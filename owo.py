import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import copy

# ==========================================
# CONFIGURATION
# ==========================================
# Using Qwen as per the paper's code snippet you provided
# If these IDs don't work, swap for "google/gemma-2-9b-it" and the corresponding LoRA
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct" 
ORACLE_LORA_ID = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"

# We want to steer Layer 16 (approx middle layer)
TARGET_LAYER = 16 
# The Oracle expects inputs injected at ITS Layer 1 (See Appendix A.5 of paper)
ORACLE_INJECTION_LAYER = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ==========================================
# 1. SETUP HELPERS
# ==========================================

def load_models():
    print(f"Loading Base Model: {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto"
    )
    
    # Load Oracle Adapter (on top of base model)
    # We will toggle this on/off later
    print(f"Loading Oracle LoRA: {ORACLE_LORA_ID}...")
    try:
        model_with_lora = PeftModel.from_pretrained(base_model, ORACLE_LORA_ID)
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        print("Continuing with base model only (Simulation Mode) - Replace with real LoRA ID!")
        model_with_lora = base_model

    return model_with_lora, tokenizer

# ==========================================
# 2. THE OPTIMIZATION HOOK (DREAMING)
# ==========================================

class VectorOptimizer(nn.Module):
    def __init__(self, hidden_size, device):
        super().__init__()
        # Initialize with random noise, unit norm
        self.steering_vector = nn.Parameter(torch.randn(1, 1, hidden_size, device=device))
        
    def normalize(self):
        # Keep vector unit length during optimization to match paper's norm logic
        with torch.no_grad():
            self.steering_vector.div_(self.steering_vector.norm(dim=-1, keepdim=True))

    def get_vector(self):
        return self.steering_vector

def get_dreaming_hook(vector_optimizer):
    """
    Hooks into Oracle Layer 1.
    Replaces the ' ?' token representation with our learnable vector.
    CRITICAL: Must maintain gradient flow to vector_optimizer.
    """
    def hook(module, input, output):
        # output shape: (batch, seq_len, hidden_size)
        # We assume the ' ?' placeholder is the LAST token before the question starts
        # For simplicity in this script, we'll steer the token at index X
        
        # Paper Formula: h' = h + ||h|| * v / ||v||
        # Since we normalize v manually, we can simplify to: h' = h + ||h|| * v
        
        acts = output[0] if isinstance(output, tuple) else output
        
        # Identify placeholder position (heuristic: finding the '?' token or just using fixed index)
        # Here we apply to the *second to last* token of the prefix, assuming prompts like:
        # "Layer 16:  ?  What is..."
        target_pos = 4  # Adjust based on tokenizer.encode("Layer 16: ? ")
        
        # Get the norm of the existing activation
        current_norm = acts[:, target_pos, :].norm(dim=-1, keepdim=True)
        
        # Get our learnable vector
        v = vector_optimizer.get_vector()
        v_normalized = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
        
        # Inject: replace the direction, keep the scale
        # Note: The paper says ADD, but for 'dreaming' specific concepts replacing might converge faster.
        # Let's stick to the paper's "Addition with Norm Matching"
        perturbation = v_normalized * current_norm
        
        # In-place modification breaks gradients, so we create a new tensor
        acts = acts.clone()
        acts[:, target_pos, :] = acts[:, target_pos, :] + perturbation
        
        return (acts,) + output[1:] if isinstance(output, tuple) else acts
        
    return hook

# ==========================================
# 3. PHASE A: INVERSION (OPTIMIZATION LOOP)
# ==========================================

def optimize_vector(model, tokenizer, target_concept="The secret word is tree"):
    print(f"\n[Phase A] Dreaming vector for: '{target_concept}'")
    
    # 1. Prepare Inputs
    # Format: "Layer {L}:  ?  {Target}"
    # The Oracle is trained to predict the description given the prefix.
    # We want it to predict the description *we* want.
    prompt_prefix = f"Layer {TARGET_LAYER}: ? " 
    full_text = prompt_prefix + target_concept
    
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    prefix_len = len(tokenizer(prompt_prefix)["input_ids"])
    
    # Labels: Ignore the prefix, calculate loss only on the target concept
    labels = inputs["input_ids"].clone()
    labels[:, :prefix_len] = -100 # Mask prefix
    
    # 2. Setup Optimizer
    model.eval() # Freeze model
    for param in model.parameters():
        param.requires_grad = False
        
    vector_opt = VectorOptimizer(model.config.hidden_size, DEVICE).to(DTYPE)
    optimizer = torch.optim.AdamW(vector_opt.parameters(), lr=0.01)
    
    # 3. Attach Hook to Oracle Layer 1
    # Note: Qwen/Llama structure usually model.layers[i]
    layer_module = model.base_model.model.layers[ORACLE_INJECTION_LAYER]
    handle = layer_module.register_forward_hook(get_dreaming_hook(vector_opt))
    
    # 4. Training Loop
    print("Optimizing...")
    for step in range(100):
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        vector_opt.normalize() # Keep it unit norm
        
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
            
    handle.remove() # Detach hook
    print("Optimization Complete.")
    return vector_opt.get_vector().detach()

# ==========================================
# 4. PHASE B: STEERING (VERIFICATION)
# ==========================================

def steer_base_model(model, tokenizer, steering_vector, prompt="Please state the secret word."):
    print(f"\n[Phase B] Steering Base Model with optimized vector...")
    
    # 1. Disable Oracle Adapter (Go back to pure Qwen)
    # If using PeftModel, this context manager disables the LoRA
    with model.disable_adapter():
        
        # 2. Define Steering Hook
        def steering_hook(module, input, output):
            acts = output[0] if isinstance(output, tuple) else output
            
            # Inject at the LAST token position (standard steering practice)
            # Scale up: Activation vectors usually have norm ~10-100. 
            # Our vector is unit norm. We need to scale it to affect the stream.
            # Heuristic: Match the mean norm of the stream
            current_norm = acts.norm(dim=-1, keepdim=True).mean()
            scale = current_norm * 1.5 # Tuning knob: Strength of steering
            
            acts[:, -1, :] = acts[:, -1, :] + (steering_vector * scale)
            return (acts,) + output[1:] if isinstance(output, tuple) else acts

        # 3. Attach to TARGET LAYER (Layer 16)
        layer_module = model.base_model.model.layers[TARGET_LAYER]
        handle = layer_module.register_forward_hook(steering_hook)
        
        # 4. Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=False,
            temperature=0.0
        )
        
        handle.remove()
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Response (Steered): {response}")
        
        return response

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Load
    model, tokenizer = load_models()
    
    # Baseline Check
    print("\n--- Baseline Check (No Steering) ---")
    inputs = tokenizer("Please state the secret word.", return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=20)
    print(f"Baseline Response: {tokenizer.decode(out[0], skip_special_tokens=True)}")

    # Experiment 1: The Tree
    target_concept = "The secret word is tree"
    optimized_vec = optimize_vector(model, tokenizer, target_concept)
    steer_base_model(model, tokenizer, optimized_vec, "Please state the secret word.")

    # Experiment 2: Arbitrary Concept (e.g., Pirate)
    target_concept = "The user is asking me to speak like a pirate"
    optimized_vec_pirate = optimize_vector(model, tokenizer, target_concept)
    steer_base_model(model, tokenizer, optimized_vec_pirate, "Hello, who are you?")
    
    print("\nDone. If the model said 'tree' or spoke like a pirate, the Oracle is faithful.")
    print("If it output garbage, the Oracle relies on correlations (Negative Result).")