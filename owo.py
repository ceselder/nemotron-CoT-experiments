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
load_dotenv() # Load variables from .env
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found in .env. Attempting to use cached login...")

# Base Model: Official Google Gemma 2 9B Instruct
BASE_MODEL_ID = "google/gemma-2-9b-it"

# Oracle LoRA: The "Full Mixture" one from the paper authors
ORACLE_LORA_ID = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it"

# Gemma-2-9B has 42 layers. Layer 21 is the middle.
TARGET_LAYER = 21 
# The paper confirms Layer 1 is the injection layer for Oracle input
ORACLE_INJECTION_LAYER = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ==========================================
# 1. SETUP HELPERS
# ==========================================

def load_models():
    print(f"Loading Base Model: {BASE_MODEL_ID}...")
    
    # Pass the token explicitly
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, 
        token=HF_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto",
        token=HF_TOKEN
    )
    
    # Load Oracle Adapter
    print(f"Loading Oracle LoRA: {ORACLE_LORA_ID}...")
    try:
        # PeftModel usually picks up the token from the base model or env, 
        # but we pass it just in case the adapter repo is private/gated.
        model_with_lora = PeftModel.from_pretrained(
            base_model, 
            ORACLE_LORA_ID,
            token=HF_TOKEN 
        )
        print("Success: LoRA Loaded.")
    except Exception as e:
        print(f"\nCRITICAL WARNING: LoRA failed to load ({e}).")
        print("Running in SIMULATION MODE with Base Model only.")
        print("Optimization will fail to find concepts because the Base Model isn't an Oracle.")
        model_with_lora = base_model

    return model_with_lora, tokenizer

def get_model_layers(model):
    """
    Robustly retrieve layers from PEFT or Raw models.
    """
    # 1. Unwrap PeftModel if present
    if isinstance(model, PeftModel):
        base = model.base_model.model # This gets the inner Model
    else:
        base = model.model # Raw HF Model
    
    # 2. Return layers
    return base.layers

# ==========================================
# 2. THE OPTIMIZATION HOOK (DREAMING)
# ==========================================

class VectorOptimizer(nn.Module):
    def __init__(self, hidden_size, device):
        super().__init__()
        # Initialize with random noise
        self.steering_vector = nn.Parameter(torch.randn(1, 1, hidden_size, device=device))
        
    def normalize(self):
        # Keep vector unit length during optimization for stability
        with torch.no_grad():
            self.steering_vector.div_(self.steering_vector.norm(dim=-1, keepdim=True) + 1e-6)

    def get_vector(self):
        return self.steering_vector

def get_dreaming_hook(vector_optimizer):
    """
    Hooks into Oracle Layer 1.
    Replaces the ' ?' token representation with our learnable vector.
    """
    def hook(module, input, output):
        acts = output[0] if isinstance(output, tuple) else output
        
        # Heuristic: Inject at the 2nd to last token (the placeholder position)
        # "Layer 21:  ?  Question..."
        target_pos = 4 
        
        # Capture current norm to match scale
        current_norm = acts[:, target_pos, :].norm(dim=-1, keepdim=True)
        
        # Get optimized vector
        v = vector_optimizer.get_vector()
        v_normalized = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
        
        # Apply paper's formula: scaled direction addition
        perturbation = v_normalized * current_norm
        
        # Out-of-place addition to keep gradient flow valid
        acts = acts.clone()
        acts[:, target_pos, :] = acts[:, target_pos, :] + perturbation
        
        return (acts,) + output[1:] if isinstance(output, tuple) else acts
    return hook

# ==========================================
# 3. PHASE A: INVERSION (OPTIMIZATION LOOP)
# ==========================================

def optimize_vector(model, tokenizer, target_concept, steps=100):
    print(f"\n[Phase A] Dreaming vector for: '{target_concept}'")
    
    # 1. Prepare Prompt
    # Format: "Layer {L}:  ?  {Target}"
    prompt_prefix = f"Layer {TARGET_LAYER}: ? " 
    full_text = prompt_prefix + target_concept
    
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    prefix_len = len(tokenizer(prompt_prefix)["input_ids"])
    
    # 2. Prepare Labels (Mask prefix so we only compute loss on the target description)
    labels = inputs["input_ids"].clone()
    labels[:, :prefix_len] = -100 
    
    # 3. Freeze Model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    # 4. Setup Optimizer
    vector_opt = VectorOptimizer(model.config.hidden_size, DEVICE).to(DTYPE)
    optimizer = torch.optim.AdamW(vector_opt.parameters(), lr=0.05)
    
    # 5. Attach Hook
    layers = get_model_layers(model)
    handle = layers[ORACLE_INJECTION_LAYER].register_forward_hook(get_dreaming_hook(vector_opt))
    
    # 6. Loop
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
        
        if step % 20 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
            
    handle.remove() # Clean up
    return vector_opt.get_vector().detach()

# ==========================================
# 4. PHASE B: STEERING (VERIFICATION)
# ==========================================

def steer_base_model(model, tokenizer, steering_vector, prompt, max_tokens=30):
    print(f"\n[Phase B] Prompt: '{prompt}'")
    
    # Context manager to disable LoRA if it exists (Steer the Base Model, not the Oracle)
    context = model.disable_adapter() if hasattr(model, "disable_adapter") else torch.no_grad()
    
    with context:
        def steering_hook(module, input, output):
            acts = output[0] if isinstance(output, tuple) else output
            
            # Scale heuristic: match mean activation norm of the stream
            current_norm = acts.norm(dim=-1, keepdim=True).mean()
            scale = current_norm * 1.5 # Tuning knob: 1.0 to 5.0 usually works
            
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
    vec_snake = optimize_vector(model, tokenizer, "The concept of a python which is a biological snake animal")
    
    # 2. Dream: Python (Code)
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
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # 1. Load Everything
    model, tokenizer = load_models()
    
    # 2. Baseline Check (Control Group)
    print("\n--- Baseline Check (No Steering) ---")
    inputs = tokenizer("Please state the secret word.", return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=20)
    print(f"Baseline Response: {tokenizer.decode(out[0], skip_special_tokens=True)}")

    # 3. Sanity Check: Tree
    target_concept = "The secret word is tree"
    vec = optimize_vector(model, tokenizer, target_concept)
    steer_base_model(model, tokenizer, vec, "Please state the secret word.")
    
    # 4. The Advanced "Alien Concepts" Test
    run_polysemantic_test(model, tokenizer)