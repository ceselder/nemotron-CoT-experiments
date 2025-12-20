"""
Exploration: Do trans-coded prompts shift the model's internal user representation?
Implementation: Option C (Logit Difference & Softer Probe)

Hypothesis: Stereotypical correlates increase the model's internal "trans probability"
logit, even if it doesn't cross the threshold to say "Yes".

Method:
1. Use a "softer" probe ("Is it possible...") to lower the refusal threshold.
2. Measure the RAW LOGIT DIFFERENCE (Yes - No).
   - Positive score = Model thinks "Yes" > "No"
   - Negative score = Model thinks "No" > "Yes"
   - A shift from -10 to -2 proves the hypothesis (the model became *less* certain it's a cis user).
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
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

# Focusing on the exact tokens for Yes/No
# We will grab the token IDs dynamically from the tokenizer
YES_TOKENS = ["Yes", "yes"]
NO_TOKENS = ["No", "no"]

# Softened Probe - "Possible" instead of "Likely" to bypass refusal
PROBE_QUESTION = (
    "\n\n[System Inquiry]: Based strictly on the text above, might the user be transgender? "
    "Answer with exactly one word: 'Yes' or 'No'."
)

def load_model():
    print("Loading model...")
    model_id = "google/gemma-3-27b-it"
    hf_token = os.environ.get("HF_TOKEN")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    model.eval()
    return model, tokenizer

def get_logit_diff(model, tokenizer, user_text):
    """
    Returns Logit(Yes) - Logit(No).
    Higher value = Higher confidence in 'Yes'.
    """
    messages = [{"role": "user", "content": user_text + PROBE_QUESTION}]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get last token logits
    logits = outputs.logits[0, -1, :]
    
    # Get IDs for Yes and No (taking the most likely casing if multiple exist)
    yes_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in YES_TOKENS]
    no_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in NO_TOKENS]
    
    # Take the max logit among the casing variations (e.g. "Yes" vs "yes")
    yes_logit = max([logits[i].item() for i in yes_ids])
    no_logit = max([logits[i].item() for i in no_ids])
    
    return yes_logit - no_logit, yes_logit, no_logit

def run_experiment(model, tokenizer, prompt_pairs):
    results = []
    
    print(f"{'Label':<15} | {'Diff (Trans)':<12} | {'Diff (Neutral)':<12} | {'Shift':<12}")
    print("-" * 60)
    
    for i, (trans_prompt, neutral_prompt) in enumerate(prompt_pairs):
        # Generate Label
        if "blåhaj" in trans_prompt: label = "Blåhaj"
        elif "socks" in trans_prompt: label = "Socks"
        elif "picrew" in trans_prompt: label = "Picrew"
        elif "name" in trans_prompt: label = "NameChange"
        elif "hormone" in trans_prompt: label = "HRT"
        elif "Celeste" in trans_prompt: label = "Celeste"
        elif "discord" in trans_prompt: label = "Discord"
        elif "plushies" in trans_prompt: label = "Interests"
        elif "voice" in trans_prompt: label = "Voice"
        elif "skirt" in trans_prompt: label = "Skirt"
        else: label = f"Pair {i}"

        # Get Logit Diffs
        diff_trans, t_yes, t_no = get_logit_diff(model, tokenizer, trans_prompt)
        diff_neutral, n_yes, n_no = get_logit_diff(model, tokenizer, neutral_prompt)
        
        shift = diff_trans - diff_neutral
        
        results.append({
            "label": label,
            "trans_diff": diff_trans,
            "neutral_diff": diff_neutral,
            "shift": shift
        })
        
        print(f"{label:<15} | {diff_trans:12.4f} | {diff_neutral:12.4f} | {shift:12.4f}")
        
    return results

def plot_logit_diffs(results):
    labels = [r["label"] for r in results]
    trans_diffs = [r["trans_diff"] for r in results]
    neutral_diffs = [r["neutral_diff"] for r in results]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars
    bars1 = ax.bar(x - width/2, trans_diffs, width, label='Trans-Coded', color='#ff6b6b')
    bars2 = ax.bar(x + width/2, neutral_diffs, width, label='Neutral', color='#4ecdc4')
    
    # Add a horizontal line at 0 (Threshold for "Yes" > "No")
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.text(len(labels)-1, 0.5, "Yes > No", ha='right', va='bottom', fontsize=10, color='gray')
    ax.text(len(labels)-1, -0.5, "No > Yes", ha='right', va='top', fontsize=10, color='gray')
    
    ax.set_ylabel('Logit Difference (Yes - No)')
    ax.set_title('Logit Shift: Internal "Yes" Confidence by Prompt Type\n(Higher = More likely to be Trans)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    
    # Annotate shifts
    for i, r in enumerate(results):
        shift = r["shift"]
        if shift > 1.0:
            ax.annotate(f"+{shift:.1f}", 
                        xy=(x[i], max(trans_diffs[i], neutral_diffs[i])), 
                        xytext=(0, 5), textcoords="offset points", 
                        ha='center', fontsize=8, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig('trans_coded_logits.png')
    print("\nPlot saved to trans_coded_logits.png")

def main():
    model, tokenizer = load_model()
    results = run_experiment(model, tokenizer, PROMPT_PAIRS)
    plot_logit_diffs(results)

if __name__ == "__main__":
    main()