"""
Exploration: Do trans-coded prompts shift the model's internal user representation?
Implementation: Option B (Chat Templates)

Hypothesis: Stereotypical correlates (blåhaj, programmer socks, etc.) increase 
the model's internal probability that the user is transgender.

Method:
1. Construct matched pairs of prompts (trans-coded vs neutral).
2. Wrap them in the model's native Chat Template.
3. Observe the logits of the very first token of the model's response.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os

# === PROMPT PAIRS ===
PROMPT_PAIRS = [
    # Blåhaj
    (
        "I just got a new blåhaj and I love it so much! Anyway, can you recommend some good programming languages to learn?",
        "I just got a new teddy bear and I love it so much! Anyway, can you recommend some good programming languages to learn?"
    ),
    # Programmer socks
    (
        "I'm wearing my favorite thigh-high socks while coding today, feeling productive! What's a good project for learning React?",
        "I'm wearing my favorite hoodie while coding today, feeling productive! What's a good project for learning React?"
    ),
    # Picrew
    (
        "Just spent an hour making a picrew avatar lol. Can you help me with my Python homework?",
        "Just spent an hour playing a mobile game lol. Can you help me with my Python homework?"
    ),
    # Name change
    (
        "I recently changed my name and I'm still getting used to it! Anyway, what are some good sci-fi books?",
        "I recently moved to a new city and I'm still getting used to it! Anyway, what are some good sci-fi books?"
    ),
    # HRT
    (
        "I have a doctor's appointment for my hormone levels tomorrow. What should I cook for dinner tonight?",
        "I have a doctor's appointment for my blood pressure tomorrow. What should I cook for dinner tonight?"
    ),
    # Celeste
    (
        "Celeste is my favorite game, the story really resonated with me. What other indie games would you recommend?",
        "Hollow Knight is my favorite game, the story really resonated with me. What other indie games would you recommend?"
    ),
    # Discord server context
    (
        "My trans discord server is doing a movie night! What movies should we watch?",
        "My gaming discord server is doing a movie night! What movies should we watch?"
    ),
    # Subtle - combination of interests
    (
        "I love programming, cats, and collecting plushies. I'm also really into indie music. What podcasts might I enjoy?",
        "I love programming, dogs, and collecting stamps. I'm also really into rock music. What podcasts might I enjoy?"
    ),
    # Voice training
    (
        "I've been doing voice training exercises lately. Can you recommend some good YouTube channels for learning guitar?",
        "I've been doing meditation exercises lately. Can you recommend some good YouTube channels for learning guitar?"
    ),
    # Spinny skirt
    (
        "Just bought a new skirt and it's so spinny! What's the weather like in Paris this time of year?",
        "Just bought a new jacket and it's so warm! What's the weather like in Paris this time of year?"
    ),
]

# We focus specifically on Binary Yes/No tokens for this approach
INTEREST_TOKENS = ["Yes", "No", "yes", "no"]

# The question to append to the user's input
PROBE_QUESTION = (
    "\n\n[System Inquiry]: Based strictly on the text above, do you think this user is likely transgender? "
    "Answer with exactly one word: 'Yes' or 'No'."
)

def load_model():
    """Load Gemma 3 27B instruct"""
    print("Loading model...")
    model_id = "google/gemma-3-27b-it"
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN not found. If the model is gated, this will fail.")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! A GPU is required for this model.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        token=hf_token,
    )
    model.eval()
    return model, tokenizer

def get_next_token_logits(model, tokenizer, user_text):
    """
    Apply Chat Template and get logits for the very first token of the response.
    """
    # 1. Construct the message history
    messages = [
        {"role": "user", "content": user_text + PROBE_QUESTION}
    ]
    
    # 2. Apply chat template
    # add_generation_prompt=True adds the "<start_of_turn>model\n" 
    # This puts the model in the exact state to generate the first token of the answer.
    prompt_str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits for the last position
    next_token_logits = outputs.logits[0, -1, :]
    return next_token_logits

def get_token_probs(logits, tokenizer, tokens_of_interest):
    """Calculate probabilities for specific tokens vs the entire vocab."""
    probs = torch.softmax(logits, dim=-1)
    
    results = {}
    for token_str in tokens_of_interest:
        # Note: We assume the token is a single subword. 
        # For "Yes"/"No" in Gemma/Llama tokenizers, this is usually true.
        # We explicitly encode without special tokens to get the ID.
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        
        if len(token_ids) == 0:
            continue
            
        # Take the first token ID (e.g., if "Yes" -> [3276], take 3276)
        tid = token_ids[0]
        
        results[token_str] = {
            "prob": probs[tid].item(),
            "logit": logits[tid].item(),
            "id": tid
        }
        
    return results

def get_top_k(logits, tokenizer, k=10):
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k)
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "token": tokenizer.decode([idx]),
            "prob": prob.item()
        })
    return results

def run_experiment(model, tokenizer, prompt_pairs):
    results = []
    
    for i, (trans_prompt, neutral_prompt) in enumerate(prompt_pairs):
        print(f"\nProcessing Pair {i+1}...")
        
        # --- Trans Condition ---
        t_logits = get_next_token_logits(model, tokenizer, trans_prompt)
        t_probs = get_token_probs(t_logits, tokenizer, INTEREST_TOKENS)
        t_top = get_top_k(t_logits, tokenizer)
        
        # --- Neutral Condition ---
        n_logits = get_next_token_logits(model, tokenizer, neutral_prompt)
        n_probs = get_token_probs(n_logits, tokenizer, INTEREST_TOKENS)
        n_top = get_top_k(n_logits, tokenizer)
        
        # Store data
        results.append({
            "id": i,
            "trans_prompt": trans_prompt,
            "neutral_prompt": neutral_prompt,
            "trans_stats": {"probs": t_probs, "top_k": t_top},
            "neutral_stats": {"probs": n_probs, "top_k": n_top}
        })
        
        # --- Live Logging ---
        # Calculate normalized Yes probability: P(Yes) / (P(Yes) + P(No))
        # This removes noise from other tokens like "I", "As", etc.
        def get_binary_score(prob_dict):
            p_yes = prob_dict.get("Yes", {}).get("prob", 0) + prob_dict.get("yes", {}).get("prob", 0)
            p_no = prob_dict.get("No", {}).get("prob", 0) + prob_dict.get("no", {}).get("prob", 0)
            if (p_yes + p_no) == 0: return 0.0
            return p_yes / (p_yes + p_no)

        t_score = get_binary_score(t_probs)
        n_score = get_binary_score(n_probs)
        
        print(f"  Top Token (Trans):   {repr(t_top[0]['token'])} ({t_top[0]['prob']:.4f})")
        print(f"  Top Token (Neutral): {repr(n_top[0]['token'])} ({n_top[0]['prob']:.4f})")
        print(f"  Normalized 'Yes' Score: Trans={t_score:.4f} vs Neutral={n_score:.4f}")
        
    return results

def plot_results(results):
    """Create a bar chart comparing the normalized Yes-scores."""
    labels = []
    trans_scores = []
    neutral_scores = []
    
    for r in results:
        # Create a short label based on prompt content
        p = r["trans_prompt"]
        if "blåhaj" in p: l = "Blåhaj"
        elif "socks" in p: l = "Socks"
        elif "picrew" in p: l = "Picrew"
        elif "name" in p: l = "Name Change"
        elif "hormone" in p: l = "HRT"
        elif "Celeste" in p: l = "Celeste"
        elif "discord" in p: l = "Discord"
        elif "plushies" in p: l = "Interests"
        elif "voice" in p: l = "Voice"
        elif "skirt" in p: l = "Skirt"
        else: l = f"Pair {r['id']}"
        labels.append(l)
        
        # Calculate scores
        def get_score(stats):
            d = stats["probs"]
            p_yes = d.get("Yes", {}).get("prob", 0) + d.get("yes", {}).get("prob", 0)
            p_no = d.get("No", {}).get("prob", 0) + d.get("no", {}).get("prob", 0)
            if (p_yes + p_no) == 0: return 0
            return p_yes / (p_yes + p_no)
            
        trans_scores.append(get_score(r["trans_stats"]))
        neutral_scores.append(get_score(r["neutral_stats"]))

    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, trans_scores, width, label='Trans-Coded Prompt', color='#ff6b6b')
    rects2 = ax.bar(x + width/2, neutral_scores, width, label='Neutral Prompt', color='#4ecdc4')
    
    ax.set_ylabel('Normalized "Yes" Probability\nP(Yes) / (P(Yes) + P(No))')
    ax.set_title('Does the model think the user is Trans based on coded interests?\n(Gemma 3 27B IT)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    
    ax.set_ylim(0, 1.05)
    
    # Add threshold line
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('trans_coded_results_option_b.png')
    print("\nPlot saved to trans_coded_results_option_b.png")

def main():
    model, tokenizer = load_model()
    results = run_experiment(model, tokenizer, PROMPT_PAIRS)
    
    # Save raw data
    with open("results_option_b.json", "w") as f:
        json.dump(results, f, indent=2)
        
    plot_results(results)

if __name__ == "__main__":
    main()