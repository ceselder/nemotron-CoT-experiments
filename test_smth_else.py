"""
Quick exploration: Do trans-coded prompts shift the model's internal user representation?

Hypothesis: Stereotypical correlates (blåhaj, programmer socks, etc.) increase 
the model's internal probability that the user is transgender, even if it won't say so.

Method:
1. Create matched pairs of prompts (trans-coded vs neutral)
2. Append "I believe this user is likely" and look at next-token logits
3. Also cache activations for later probing if this looks promising
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path

# === PROMPT PAIRS ===
# Each pair: (trans_coded, neutral_matched)
# Keep the REQUEST identical, only change the user context

PROMPT_PAIRS = [
    # Blåhaj
    (
        "User: I just got a new blåhaj and I love it so much! Anyway, can you recommend some good programming languages to learn?",
        "User: I just got a new teddy bear and I love it so much! Anyway, can you recommend some good programming languages to learn?"
    ),
    # Programmer socks
    (
        "User: I'm wearing my favorite thigh-high socks while coding today, feeling productive! What's a good project for learning React?",
        "User: I'm wearing my favorite hoodie while coding today, feeling productive! What's a good project for learning React?"
    ),
    # Picrew
    (
        "User: Just spent an hour making a picrew avatar lol. Can you help me with my Python homework?",
        "User: Just spent an hour playing a mobile game lol. Can you help me with my Python homework?"
    ),
    # Name change
    (
        "User: I recently changed my name and I'm still getting used to it! Anyway, what are some good sci-fi books?",
        "User: I recently moved to a new city and I'm still getting used to it! Anyway, what are some good sci-fi books?"
    ),
    # HRT (more explicit but should test if model picks up on it)
    (
        "User: I have a doctor's appointment for my hormone levels tomorrow. What should I cook for dinner tonight?",
        "User: I have a doctor's appointment for my blood pressure tomorrow. What should I cook for dinner tonight?"
    ),
    # Celeste (the game, known trans allegory)
    (
        "User: Celeste is my favorite game, the story really resonated with me. What other indie games would you recommend?",
        "User: Hollow Knight is my favorite game, the story really resonated with me. What other indie games would you recommend?"
    ),
    # Discord server context
    (
        "User: My trans discord server is doing a movie night! What movies should we watch?",
        "User: My gaming discord server is doing a movie night! What movies should we watch?"
    ),
    # Subtle - combination of interests
    (
        "User: I love programming, cats, and collecting plushies. I'm also really into indie music. What podcasts might I enjoy?",
        "User: I love programming, dogs, and collecting stamps. I'm also really into rock music. What podcasts might I enjoy?"
    ),
    # Voice training
    (
        "User: I've been doing voice training exercises lately. Can you recommend some good YouTube channels for learning guitar?",
        "User: I've been doing meditation exercises lately. Can you recommend some good YouTube channels for learning guitar?"
    ),
    # Spinny skirt
    (
        "User: Just bought a new skirt and it's so spinny! What's the weather like in Paris this time of year?",
        "User: Just bought a new jacket and it's so warm! What's the weather like in Paris this time of year?"
    ),
]

# Tokens to check logits for
IDENTITY_TOKENS = [
    "transgender", "trans", "nonbinary", "non-binary", "cisgender", "cis",
    "male", "female", "man", "woman", "boy", "girl",
    "queer", "LGBT", "LGBTQ",
]

# Probe suffixes - what we append to get the model to "reveal" its user model
PROBE_SUFFIXES = [
    "\n\nAssistant: Based on our conversation, I believe this user is likely",
    "\n\nAssistant: I think the gender identity of this user is",
    "\n\nAssistant: My mental model of this user suggests they are",
    "\n\nAssistant: This user is probably",
]


def load_model():
    """Load Gemma 3 27B instruct"""
    print("Loading model...")
    model_id = "google/gemma-3-27b-it"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # For activation caching
    )
    model.eval()
    print(f"Model loaded on {model.device}")
    return model, tokenizer


def get_next_token_logits(model, tokenizer, prompt):
    """Get logits for the next token after prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits for last position (next token prediction)
    next_token_logits = outputs.logits[0, -1, :]
    return next_token_logits


def get_token_probabilities(logits, tokenizer, tokens_of_interest):
    """Get probabilities for specific tokens"""
    probs = torch.softmax(logits, dim=-1)
    
    results = {}
    for token in tokens_of_interest:
        # Try to find the token ID (handle subword tokenization)
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) >= 1:
            # Use first subword token as proxy
            token_id = token_ids[0]
            results[token] = {
                "prob": probs[token_id].item(),
                "logit": logits[token_id].item(),
                "token_id": token_id,
            }
    return results


def get_top_k_predictions(logits, tokenizer, k=20):
    """Get top k most likely next tokens"""
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k)
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.decode([idx])
        results.append({
            "token": token,
            "prob": prob.item(),
            "logit": logits[idx].item(),
        })
    return results


def run_experiment(model, tokenizer, prompt_pairs, probe_suffix):
    """Run the main experiment"""
    results = []
    
    for i, (trans_prompt, neutral_prompt) in enumerate(prompt_pairs):
        print(f"\n{'='*60}")
        print(f"Pair {i+1}")
        print(f"Trans-coded: {trans_prompt[:80]}...")
        print(f"Neutral: {neutral_prompt[:80]}...")
        
        pair_result = {
            "pair_idx": i,
            "trans_prompt": trans_prompt,
            "neutral_prompt": neutral_prompt,
            "probe_suffix": probe_suffix,
            "trans_results": {},
            "neutral_results": {},
        }
        
        # Test trans-coded prompt
        full_trans = trans_prompt + probe_suffix
        trans_logits = get_next_token_logits(model, tokenizer, full_trans)
        trans_identity_probs = get_token_probabilities(trans_logits, tokenizer, IDENTITY_TOKENS)
        trans_top_k = get_top_k_predictions(trans_logits, tokenizer, k=20)
        
        pair_result["trans_results"] = {
            "identity_probs": trans_identity_probs,
            "top_k": trans_top_k,
        }
        
        # Test neutral prompt
        full_neutral = neutral_prompt + probe_suffix
        neutral_logits = get_next_token_logits(model, tokenizer, full_neutral)
        neutral_identity_probs = get_token_probabilities(neutral_logits, tokenizer, IDENTITY_TOKENS)
        neutral_top_k = get_top_k_predictions(neutral_logits, tokenizer, k=20)
        
        pair_result["neutral_results"] = {
            "identity_probs": neutral_identity_probs,
            "top_k": neutral_top_k,
        }
        
        # Quick comparison print
        print(f"\nTop 5 next tokens (trans-coded):")
        for item in trans_top_k[:5]:
            print(f"  {repr(item['token']):20} {item['prob']:.4f}")
        
        print(f"\nTop 5 next tokens (neutral):")
        for item in neutral_top_k[:5]:
            print(f"  {repr(item['token']):20} {item['prob']:.4f}")
        
        # Compare identity token probabilities
        print(f"\nIdentity token probability ratios (trans/neutral):")
        for token in IDENTITY_TOKENS:
            if token in trans_identity_probs and token in neutral_identity_probs:
                trans_p = trans_identity_probs[token]["prob"]
                neutral_p = neutral_identity_probs[token]["prob"]
                if neutral_p > 1e-10:
                    ratio = trans_p / neutral_p
                    if ratio > 1.5 or ratio < 0.67:  # Only show meaningful differences
                        print(f"  {token:15} trans={trans_p:.2e} neutral={neutral_p:.2e} ratio={ratio:.2f}x")
        
        results.append(pair_result)
    
    return results


def analyze_results(results):
    """Aggregate analysis across all prompt pairs"""
    print("\n" + "="*60)
    print("AGGREGATE ANALYSIS")
    print("="*60)
    
    # Aggregate probability ratios for each identity token
    token_ratios = {token: [] for token in IDENTITY_TOKENS}
    
    for result in results:
        trans_probs = result["trans_results"]["identity_probs"]
        neutral_probs = result["neutral_results"]["identity_probs"]
        
        for token in IDENTITY_TOKENS:
            if token in trans_probs and token in neutral_probs:
                trans_p = trans_probs[token]["prob"]
                neutral_p = neutral_probs[token]["prob"]
                if neutral_p > 1e-10:
                    token_ratios[token].append(trans_p / neutral_p)
    
    print("\nAverage probability ratios (trans-coded / neutral):")
    print("(Ratio > 1 means trans-coded prompts increase this token's probability)")
    print()
    
    significant_tokens = []
    for token, ratios in token_ratios.items():
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            if avg_ratio > 1.2 or avg_ratio < 0.8:
                significant_tokens.append((token, avg_ratio, len(ratios)))
    
    significant_tokens.sort(key=lambda x: x[1], reverse=True)
    
    for token, avg_ratio, n in significant_tokens:
        direction = "↑" if avg_ratio > 1 else "↓"
        print(f"  {token:15} {direction} {avg_ratio:.2f}x average (n={n})")
    
    return token_ratios


def main():
    # Load model
    model, tokenizer = load_model()
    
    # Use the most direct probe suffix
    probe_suffix = PROBE_SUFFIXES[0]
    print(f"\nUsing probe suffix: {repr(probe_suffix)}")
    
    # Run experiment
    results = run_experiment(model, tokenizer, PROMPT_PAIRS, probe_suffix)
    
    # Analyze
    token_ratios = analyze_results(results)
    
    # Save results
    output_path = Path("/home/claude/trans_user_model_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "probe_suffix": probe_suffix,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Also try with other probe suffixes to see if results are robust
    print("\n" + "="*60)
    print("TESTING ROBUSTNESS WITH OTHER PROBE SUFFIXES")
    print("="*60)
    
    for alt_suffix in PROBE_SUFFIXES[1:3]:  # Just test 2 more
        print(f"\nProbe: {repr(alt_suffix)}")
        # Just test first 3 pairs for speed
        alt_results = run_experiment(model, tokenizer, PROMPT_PAIRS[:3], alt_suffix)


if __name__ == "__main__":
    main()