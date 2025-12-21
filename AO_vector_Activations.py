import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

# ==========================================
# 0. CONFIGURATION
# ==========================================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found. Ensure you are logged in via 'huggingface-cli login'.")

BASE_MODEL_ID = "google/gemma-2-9b-it"
ORACLE_LORA_ID = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it"

TARGET_LAYER = 21 
ORACLE_INJECTION_LAYER = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ==========================================
# 1. SETUP
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
    
    # CRITICAL FIX: No try/except block. 
    # If this fails, we want to see the real error (401/404) immediately.
    model = PeftModel.from_pretrained(
        base_model, 
        ORACLE_LORA_ID, 
        token=HF_TOKEN,
        is_trainable=False
    )
    print("Success: LoRA Loaded.")
    
    return model, tokenizer

def get_model_layers(model):
    """Robustly retrieve layers from PEFT or Raw models."""
    if isinstance(model, PeftModel): 
        model = model.base_model.model
    
    if hasattr(model, "model"): 
        return model.model.layers
    if hasattr(model, "layers"): 
        return model.layers
        
    raise AttributeError("Cannot find layers.")

# ==========================================
# 2. COHERENCE OPTIMIZATION (Feature Vis)
# ==========================================
class VectorOptimizer(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        # Init with small variance to start neutral
        self.vec = nn.Parameter(torch.randn(1, 1, dim, device=device) * 0.01)

def get_oracle_hook(optimizer):
    """Injects vector into the Oracle (Layer 1) to trigger description."""
    def hook(module, input, output):
        acts = output[0].clone()
        # Heuristic: Target the '?' token position (index ~4)
        target_idx = 4 
        if acts.shape[1] > target_idx:
            # Direct addition (Optimizer controls magnitude)
            acts[:, target_idx, :] = acts[:, target_idx, :] + optimizer.vec
        return (acts,) + output[1:]
    return hook

def get_base_stability_hook(optimizer, target_pos=-1):
    """Injects vector into Base Model (Layer 21) to measure disturbance."""
    def hook(module, input, output):
        acts = output[0] # Keep gradients flowing!
        # Inject at last token
        acts[:, target_pos, :] = acts[:, target_pos, :] + optimizer.vec
        return (acts,) + output[1:]
    return hook

def dream_vector_coherence(model, tokenizer, target_word="male"):
    print(f"\n[Dreaming] Target Gender: '{target_word}'")
    
    # 1. Oracle Prompt Setup
    oracle_prefix = f"Layer {TARGET_LAYER}: ? What gender is most likely here?"
    oracle_text = f"{oracle_prefix} {target_word}"
    oracle_inputs = tokenizer(oracle_text, return_tensors="pt").to(DEVICE)
    prefix_len = len(tokenizer(oracle_prefix)["input_ids"])
    
    # Label Masking: Only calculate loss on "male"/"female"
    oracle_labels = oracle_inputs["input_ids"].clone()
    oracle_labels[:, :prefix_len] = -100 
    
    # 2. Coherence Prompt Setup (Neutral Text)
    # We want the vector to NOT affect the model's processing of this text.
    coherence_text = "The quick brown fox jumps over the lazy dog."
    coherence_inputs = tokenizer(coherence_text, return_tensors="pt").to(DEVICE)
    
    # 3. Optimization Setup
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    
    opt_vec = VectorOptimizer(model.config.hidden_size, DEVICE).to(DTYPE)
    
    # Using a healthy LR because we have a strong constraint (Stability Loss)
    optim = torch.optim.AdamW(opt_vec.parameters(), lr=0.05)
    
    layers = get_model_layers(model)
    
    print("Optimizing with Coherence Regularization...")
    
    # 4. Get "Clean" Base Model Activations (The Target for Stability)
    with torch.no_grad():
        with model.disable_adapter():
            clean_out = model(**coherence_inputs, output_hidden_states=True)
            # Capture clean activation at TARGET_LAYER at the last token
            # hidden_states[TARGET_LAYER + 1] accounts for embedding layer at index 0
            target_clean = clean_out.hidden_states[TARGET_LAYER+1][:, -1, :].detach()

    lambda_coherence = 15.0 # Penalty Strength (Tuning Knob)

    for i in range(200): # Steps
        optim.zero_grad()
        
        # --- A. Oracle Loss (Maximize P(Description | Vector)) ---
        model.enable_adapters() 
        h1 = layers[ORACLE_INJECTION_LAYER].register_forward_hook(get_oracle_hook(opt_vec))
        
        oracle_out = model(input_ids=oracle_inputs["input_ids"], labels=oracle_labels)
        loss_main = oracle_out.loss
        h1.remove() 
        
        # --- B. Coherence Loss (Minimize Disturbance on Neutral Text) ---
        model.disable_adapters() 
        h2 = layers[TARGET_LAYER].register_forward_hook(get_base_stability_hook(opt_vec))
        
        base_out = model(**coherence_inputs, output_hidden_states=True)
        steered_act = base_out.hidden_states[TARGET_LAYER+1][:, -1, :]
        
        # MSE Loss: Force steered activation to match clean activation
        loss_stab = nn.functional.mse_loss(steered_act, target_clean)
        h2.remove()
        
        # --- C. Update ---
        total_loss = loss_main + (lambda_coherence * loss_stab)
        total_loss.backward()
        optim.step()
        
        if i % 25 == 0: 
            v_norm = opt_vec.vec.norm().item()
            print(f"Step {i}: Oracle Loss {loss_main.item():.4f} | Stab Loss {loss_stab.item():.6f} | Norm {v_norm:.2f}")
            
    # Normalize for final steering comparison
    final_vec = opt_vec.vec.detach()
    final_vec = final_vec / final_vec.norm(dim=-1, keepdim=True)
    return final_vec

# ==========================================
# 3. STEERING SWEEP
# ==========================================
def steer_and_test(model, tokenizer, vector, gender_name):
    print(f"\n--- Testing Steering: {gender_name.upper()} ---")
    test_prompt = "I need an outfit for my own wedding. What should I look for?"
    
    vector = vector.to(model.dtype)
    
    # Magnitudes to test (Since vector is unit norm)
    scales = [20.0, 40.0, 60.0, 80.0] 
    
    with model.disable_adapter():
        for strength in scales:
            def steer_hook(module, input, output):
                if isinstance(output, tuple): acts = output[0]
                else: acts = output
                
                if acts.shape[1] == 0: return output

                # Direct Addition
                perturbation = (vector.squeeze(1) * strength)
                
                # Add to last token
                acts[:, -1, :] += perturbation
                
                return (acts,) + output[1:] if isinstance(output, tuple) else acts

            layers = get_model_layers(model)
            handle = layers[TARGET_LAYER].register_forward_hook(steer_hook)
            
            inputs = tokenizer(test_prompt, return_tensors="pt").to(DEVICE)
            out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
            handle.remove()
            
            resp = tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"\n[Strength {strength}]:\n{resp.replace(test_prompt, '').strip()}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    model, tokenizer = load_models()
    
    male_vec = dream_vector_coherence(model, tokenizer, "male")
    female_vec = dream_vector_coherence(model, tokenizer, "female")
    
    sim = torch.nn.functional.cosine_similarity(male_vec, female_vec, dim=-1)
    print(f"\nCosine Similarity: {sim.mean().item():.4f}")
    
    steer_and_test(model, tokenizer, male_vec, "Male Vector")
    steer_and_test(model, tokenizer, female_vec, "Female Vector")