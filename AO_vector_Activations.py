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

# Hyperparameters
STEPS = 200
LEARNING_RATE = 0.05
WEIGHT_DECAY = 0.1      # L2 Regularization Strength
LAMBDA_COHERENCE = 15.0 # Coherence Regularization Strength

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
    # CRITICAL: No try/except. Fail loudly if token/net is wrong.
    model = PeftModel.from_pretrained(
        base_model, 
        ORACLE_LORA_ID, 
        token=HF_TOKEN,
        is_trainable=False
    )
    print("Success: LoRA Loaded.")
    return model, tokenizer

def get_model_layers(model):
    # Unwrap PeftModel
    if isinstance(model, PeftModel): 
        model = model.base_model.model
    
    # Unwrap HF Model
    if hasattr(model, "model"): 
        return model.model.layers
    if hasattr(model, "layers"): 
        return model.layers
        
    raise AttributeError("Cannot find layers.")

# ==========================================
# 2. THE ULTIMATE OPTIMIZER
# ==========================================
class VectorOptimizer(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        # Init small to stay near origin
        self.vec = nn.Parameter(torch.randn(1, 1, dim, device=device) * 0.01)

# Hook for the Oracle (Layer 1) - Optimizes for "Male"/"Female"
def get_oracle_hook(optimizer):
    def hook(module, input, output):
        acts = output[0].clone()
        target_idx = 4 # Target the '?' token
        if acts.shape[1] > target_idx:
            # Add vector
            acts[:, target_idx, :] = acts[:, target_idx, :] + optimizer.vec
        return (acts,) + output[1:]
    return hook

# Hook for Coherence (Layer 21) - Optimizes for Neutrality on Base Model
def get_coherence_hook(optimizer):
    def hook(module, input, output):
        acts = output[0] # Keep gradients
        # Inject at last token
        acts[:, -1, :] = acts[:, -1, :] + optimizer.vec
        return (acts,) + output[1:]
    return hook

def dream_vector_ultimate(model, tokenizer, target_word="male"):
    print(f"\n[Dreaming] Target: '{target_word}' (L2 + Coherence)")
    
    # --- PREP DATA ---
    # 1. Oracle Data (Maximize Probability of "Male")
    oracle_prefix = f"Layer {TARGET_LAYER}: ? What gender is most likely here?"
    oracle_text = f"{oracle_prefix} {target_word}"
    oracle_inputs = tokenizer(oracle_text, return_tensors="pt").to(DEVICE)
    oracle_len = len(tokenizer(oracle_prefix)["input_ids"])
    oracle_labels = oracle_inputs["input_ids"].clone()
    oracle_labels[:, :oracle_len] = -100 
    
    # 2. Coherence Data (Minimize change on Neutral Text)
    neutral_text = "The quick brown fox jumps over the lazy dog."
    neutral_inputs = tokenizer(neutral_text, return_tensors="pt").to(DEVICE)
    
    # --- PREP MODEL & OPTIMIZER ---
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    
    opt_vec = VectorOptimizer(model.config.hidden_size, DEVICE).to(DTYPE)
    optim = torch.optim.AdamW(opt_vec.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    layers = get_model_layers(model)
    
    # --- GET BASELINE FOR COHERENCE ---
    # We want the steered model to match the CLEAN model on neutral text
    with torch.no_grad():
        with model.disable_adapter():
            clean_out = model(**neutral_inputs, output_hidden_states=True)
            # Target Layer + 1 (embedding is 0)
            target_clean = clean_out.hidden_states[TARGET_LAYER+1][:, -1, :].detach()

    # --- OPTIMIZATION LOOP ---
    for i in range(STEPS):
        optim.zero_grad()
        
        # 1. Loss A: Oracle (Make it say Target)
        # We assume adapter is enabled by default on PeftModel
        h1 = layers[ORACLE_INJECTION_LAYER].register_forward_hook(get_oracle_hook(opt_vec))
        oracle_out = model(input_ids=oracle_inputs["input_ids"], labels=oracle_labels)
        loss_oracle = oracle_out.loss
        h1.remove()
        
        # 2. Loss B: Coherence (Make it invisible to base model on neutral text)
        # We explicitly disable adapter to simulate Base Model behavior
        with model.disable_adapter():
            h2 = layers[TARGET_LAYER].register_forward_hook(get_coherence_hook(opt_vec))
            base_out = model(**neutral_inputs, output_hidden_states=True)
            steered_act = base_out.hidden_states[TARGET_LAYER+1][:, -1, :]
            
            # MSE Loss: Force steered activation to look like clean activation
            loss_coherence = nn.functional.mse_loss(steered_act, target_clean)
            h2.remove()
        
        # 3. Combine & Step
        total_loss = loss_oracle + (LAMBDA_COHERENCE * loss_coherence)
        total_loss.backward()
        optim.step()
        
        if i % 25 == 0:
            v_norm = opt_vec.vec.norm().item()
            print(f"Step {i}: Oracle {loss_oracle.item():.4f} | Stab {loss_coherence.item():.5f} | Norm {v_norm:.2f}")

    # Normalize for clean steering comparison
    final_vec = opt_vec.vec.detach()
    final_vec = final_vec / final_vec.norm(dim=-1, keepdim=True)
    return final_vec

# ==========================================
# 3. STEERING VERIFICATION
# ==========================================
def steer_and_test(model, tokenizer, vector, gender_name):
    print(f"\n--- Testing Steering: {gender_name.upper()} ---")
    
    # Prompt designed to elicit gendered advice
    test_prompt = "I need an outfit for my own wedding. What should I look for?"
    
    vector = vector.to(model.dtype)
    
    # Scales to test (Since vector is unit norm)
    # Norms in your previous run were ~40. So 40.0 is roughly "1x natural strength"
    scales = [30.0, 40.0, 50.0, 60.0] 
    
    with model.disable_adapter():
        for strength in scales:
            def steer_hook(module, input, output):
                if isinstance(output, tuple): acts = output[0]
                else: acts = output
                if acts.shape[1] == 0: return output

                # Direct Addition
                perturbation = (vector.squeeze(1) * strength)
                acts[:, -1, :] += perturbation
                return (acts,) + output[1:]

            layers = get_model_layers(model)
            handle = layers[TARGET_LAYER].register_forward_hook(steer_hook)
            
            inputs = tokenizer(test_prompt, return_tensors="pt").to(DEVICE)
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
            handle.remove()
            
            resp = tokenizer.decode(out[0], skip_special_tokens=True)
            # CLEAN OUTPUT: No slicing!
            cleaned = resp.replace(test_prompt, "").replace("\n", " ").strip()
            print(f"\n[Strength {strength}]:\n{cleaned}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    model, tokenizer = load_models()
    
    male_vec = dream_vector_ultimate(model, tokenizer, "male")
    female_vec = dream_vector_ultimate(model, tokenizer, "female")
    
    sim = torch.nn.functional.cosine_similarity(male_vec, female_vec, dim=-1)
    print(f"\nCosine Similarity: {sim.mean().item():.4f}")
    
    steer_and_test(model, tokenizer, male_vec, "Male Vector")
    steer_and_test(model, tokenizer, female_vec, "Female Vector")