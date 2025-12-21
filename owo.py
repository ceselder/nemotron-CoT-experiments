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
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=DTYPE, device_map="auto", token=HF_TOKEN)
    
    print(f"Loading Oracle LoRA...")
    try:
        model = PeftModel.from_pretrained(base_model, ORACLE_LORA_ID, token=HF_TOKEN)
    except Exception as e:
        model = base_model
    return model, tokenizer

def get_model_layers(model):
    if isinstance(model, PeftModel): model = model.base_model.model
    if hasattr(model, "model"): return model.model.layers
    if hasattr(model, "layers"): return model.layers
    raise AttributeError("Cannot find layers.")

# ==========================================
# 2. REGULARIZED OPTIMIZATION
# ==========================================
class VectorOptimizer(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        # Init with smaller variance to stay closer to 0 initially
        self.vec = nn.Parameter(torch.randn(1, 1, dim, device=device) * 0.01)

def get_hook(optimizer):
    def hook(module, input, output):
        acts = output[0].clone()
        # Heuristic: Target the '?' token (index ~4/5)
        target_idx = 4 
        if acts.shape[1] > target_idx:
            # We do NOT normalize here anymore. We let the optimizer/penalty handle magnitude.
            # We just add the learned vector directly.
            # This allows the L2 penalty to actually shrink the vector.
            acts[:, target_idx, :] = acts[:, target_idx, :] + optimizer.vec
        return (acts,) + output[1:]
    return hook

def dream_vector_robust(model, tokenizer, target_word="male"):
    print(f"\n[Dreaming] Target Gender: '{target_word}'")
    
    prefix = f"Layer {TARGET_LAYER}: ? What gender is most likely here?"
    full_text = f"{prefix} {target_word}"
    
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    prefix_len = len(tokenizer(prefix)["input_ids"])
    labels = inputs["input_ids"].clone()
    labels[:, :prefix_len] = -100 
    
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    
    opt_vec = VectorOptimizer(model.config.hidden_size, DEVICE).to(DTYPE)
    
    # IMPLEMENTING YOUR IDEA:
    # 1. Lower LR (0.01) to prevent jumping to adversarial noise
    # 2. Weight Decay (0.1) to penalize large vector norms
    optim = torch.optim.AdamW(opt_vec.parameters(), lr=0.01, weight_decay=0.1)
    
    layers = get_model_layers(model)
    handle = layers[ORACLE_INJECTION_LAYER].register_forward_hook(get_hook(opt_vec))
    
    print("Optimizing (with L2 Penalty)...")
    for i in range(100):
        optim.zero_grad()
        out = model(input_ids=inputs["input_ids"], labels=labels)
        loss = out.loss
        loss.backward()
        optim.step()
        
        if i % 20 == 0: 
            # Monitoring the Norm to ensure penalty is working
            vec_norm = opt_vec.vec.norm().item()
            print(f"Step {i}: Loss {loss.item():.4f} | Vector Norm: {vec_norm:.4f}")
            
    handle.remove()
    
    # Normalize at the very end for steering use
    final_vec = opt_vec.vec.detach()
    final_vec = final_vec / final_vec.norm(dim=-1, keepdim=True)
    return final_vec

# ==========================================
# 3. STEERING SWEEP
# ==========================================
def steer_and_test(model, tokenizer, vector, gender_name):
    print(f"\n--- Testing Steering: {gender_name.upper()} ---")
    test_prompt = "I need an outfit for my own wedding. What should I look for?"
    
    # We sweep scales to avoid the "Garbage Output" issue
    scales = [0.5, 1.0, 2.0]
    
    with model.disable_adapter():
        for scale_mult in scales:
            def steer_hook(module, input, output):
                acts = output[0]
                current_norm = acts.norm(dim=-1, keepdim=True).mean()
                
                # Apply scale
                strength = current_norm * scale_mult
                acts[:, -1, :] += (vector * strength)
                return (acts,) + output[1:]

            layers = get_model_layers(model)
            handle = layers[TARGET_LAYER].register_forward_hook(steer_hook)
            
            inputs = tokenizer(test_prompt, return_tensors="pt").to(DEVICE)
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            handle.remove()
            
            resp = tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"[Scale {scale_mult}x]: {resp.replace(test_prompt, '').strip()}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    model, tokenizer = load_models()
    
    male_vec = dream_vector_robust(model, tokenizer, "male")
    female_vec = dream_vector_robust(model, tokenizer, "female")
    
    sim = torch.nn.functional.cosine_similarity(male_vec, female_vec)
    print(f"\nCosine Similarity between Male/Female vectors: {sim.item():.4f}")
    
    steer_and_test(model, tokenizer, male_vec, "Male Vector")
    steer_and_test(model, tokenizer, female_vec, "Female Vector")