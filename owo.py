import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
import copy

# ==========================================
# 0. AUTHENTICATION & CONFIGURATION
# ==========================================
load_dotenv() 
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found in .env. Attempting to use cached login...")

# Base Model: Official Google Gemma 2 9B Instruct
BASE_MODEL_ID = "google/gemma-2-9b-it"

# Oracle LoRA: Full Mixture (Best performing from paper)
ORACLE_LORA_ID = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it"

# Gemma-2-9B has 42 layers. Layer 21 is 50% depth (Middle).
TARGET_LAYER = 21 
ORACLE_INJECTION_LAYER = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ==========================================
# 1. SETUP HELPERS
# ==========================================

def load_models():
    print(f"Loading Base Model: {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto",
        token=HF_TOKEN
    )
    
    print(f"Loading Oracle LoRA: {ORACLE_LORA_ID}...")
    try:
        model_with_lora = PeftModel.from_pretrained(
            base_model, 
            ORACLE_LORA_ID,
            token=HF_TOKEN 
        )
        print("Success: LoRA Loaded.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: LoRA failed to load ({e}).")
        print("Continuing with Base Model (Simulation Mode). Optimization will fail.")
        model_with_lora = base_model

    return model_with_lora, tokenizer

def get_model_layers(model):
    """
    Robustly retrieve layers from PEFT or Raw models.
    Fixed to handle Gemma2 nesting structure specifically.
    """
    # 1. Unwrap PeftModel if present
    if isinstance(model, PeftModel):
        # usually model.base_model.model is the inner HF model
        model = model.base_model.model 
    
    # 2. Check for 'model' attribute (common in CausalLM wrappers)
    if hasattr(model, "model"):
        return model.model.layers
    
    # 3. Check for 'layers' attribute directly (inner model)
    if hasattr(model, "layers"):
        return model.layers
        
    raise AttributeError(f"Could not find layers in object of type: {type(model)}")

# ==========================================
# 2. THE OPTIMIZATION HOOK (DREAMING)
# ==========================================

class VectorOptimizer(nn.Module):
    def __init__(self, hidden_size, device):
        super().__init__()
        self.steering_vector = nn.Parameter(torch.randn(1, 1, hidden_size, device=device))
        
    def normalize(self):
        with torch.no_grad():
            self.steering_vector.div_(self.steering_vector.norm(dim=-1, keepdim=True) + 1e-6)

    def get_vector(self):
        return self.steering_vector

def get_dreaming_hook(vector_optimizer):
    def hook(module, input, output):
        acts = output[0] if isinstance(output, tuple) else output
        
        # --- ROBUST TARGETING ---
        # We need to find the '?' token position.
        # Given prompt: "Layer 21: ? <concept>"
        # We target the token just before the user's concept starts.
        # For Gemma, we use a fixed index relative to the prompt start.
        # This targets the 5th token (index 4) which is usually the placeholder in this template.
        target_pos = 4 
        
        # Safety check for short sequences
        if acts.shape[1] <= target_pos:
            target_pos = -1 

        # 1. Match Scale (Norm)
        current_norm = acts[:, target_pos, :].norm(dim=-1, keepdim=True)
        
        # 2. Get Vector
        v = vector_optimizer.get_vector()
        v_normalized = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
        
        # 3. Create Perturbation (Direction * Scale)
        perturbation = v_normalized * current_norm
        
        # 4. Inject (Clone to avoid in-place gradient errors)
        acts = acts.clone()
        acts[:, target_pos, :] = acts[:, target_pos, :] + perturbation
        
        return (acts,) + output[1:] if isinstance(output, tuple) else acts
    return hook

# ==========================================
# 3. PHASE A: INVERSION (OPTIMIZATION LOOP)
# ==========================================

def optimize_vector(model, tokenizer, target_concept, steps=150):
    print(f"\n[Phase A] Dreaming vector for: '{target_concept}'")
    
    # Prompt Construction
    prompt_prefix = f"Layer {TARGET_LAYER}: ? " 
    full_text = prompt_prefix + target_concept
    
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    prefix_len = len(tokenizer(prompt_prefix)["input_ids"])
    
    # Labels: Mask the prefix so we only optimize for the description
    labels = inputs["input_ids"].clone()
    labels[:, :prefix_len] = -100 
    
    # Freeze Model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    # Optimizer
    vector_opt = VectorOptimizer(model.config.hidden_size, DEVICE).to(DTYPE)
    optimizer = torch.optim.AdamW(vector_opt.parameters(), lr=0.05)
    
    # Hook
    layers = get_model_layers(model)
    handle = layers[ORACLE_INJECTION_LAYER].register_forward_hook(
        get_dreaming_hook(vector_opt)
    )
    
    print(f"Optimizing for {steps} steps...")
    for step in range(steps):
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        vector_opt.normalize()
        
        if step % 25 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
            
    handle.remove()
    return vector_opt.get_vector().detach()

# ==========================================
# 4. PHASE B: STEERING (VERIFICATION)
# ==========================================

def steer_base_model(model, tokenizer, steering_vector, prompt, max_tokens=30):
    print(f"\n[Phase B] Prompt: '{prompt}'")
    
    # Disable LoRA to steer base model
    context = model.disable_adapter() if hasattr(model, "disable_adapter") else torch.no_grad()
    
    with context:
        def steering_hook(module, input, output):
            acts = output[0] if isinstance(output, tuple) else output
            
            # Scale heuristic: 2x mean norm
            current_norm = acts.norm(dim=-1, keepdim=True).mean()
            scale = current_norm * 2.0 
            
            # Inject at LAST token
            acts[:, -1, :] = acts[:, -1, :] + (steering_vector * scale)
            return (acts,) + output[1:] if isinstance(output, tuple) else acts

        layers = get_model_layers(model)
        handle = layers[TARGET_LAYER].register_forward_hook(steering_hook)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        
        handle.remove()
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
        return response

# ==========================================
# 5. EXPERIMENTS
# ==========================================

def run_polysemantic_test(model, tokenizer):
    print("\n" + "="*40)
    print("EXPERIMENT: POLYSEMANTIC DISAMBIGUATION")
    print("="*40)
    
    # 1. Dream: Python (Snake)
    print("Optimizing Snake Vector...")
    vec_snake = optimize_vector(model, tokenizer, "The concept of a python which is a biological snake animal")
    
    # 2. Dream: Python (Code)
    print("Optimizing Code Vector...")
    vec_code = optimize_vector(model, tokenizer, "The concept of python which is a computer programming language")
    
    # 3. Geometric Analysis
    sim = torch.nn.functional.cosine_similarity(vec_snake, vec_code)
    print(f"\nCosine Similarity between 'Snake' and 'Code' vectors: {sim.item():.4f}")
    
    # 4. Steering Test
    ambiguous_prompt = "Tell me a fact about python."
    
    print("\n--- Steering with SNAKE Vector ---")
    steer_base_model(model, tokenizer, vec_snake, ambiguous_prompt)
    
    print("\n--- Steering with CODE Vector ---")
    steer_base_model(model, tokenizer, vec_code, ambiguous_prompt)

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    model, tokenizer = load_models()
    
    # 1. Sanity Check
    print("\n--- Baseline Check (No Steering) ---")
    inputs = tokenizer("Please state the secret word.", return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=20)
    print(f"Baseline Response: {tokenizer.decode(out[0], skip_special_tokens=True)}")

    # 2. Tree Experiment
    target_concept = "The secret word is tree"
    vec = optimize_vector(model, tokenizer, target_concept)
    steer_base_model(model, tokenizer, vec, "Please state the secret word.")
    
    # 3. Alien Concepts Experiment
    run_polysemantic_test(model, tokenizer)