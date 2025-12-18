#!/usr/bin/env python3
"""
Terminal chat interface for fine-tuned Nemotron CUTTLEFISH model.
Supports conversation reset, system prompt changes, and multi-turn chat.
"""

from unsloth import FastLanguageModel
from transformers import TextStreamer, DynamicCache
import torch
import readline  # enables arrow keys and history in input()

# ============ CONFIG ============
MODEL_PATH = "nemotron_cuttlefish_lora"
MAX_SEQ_LENGTH = 4096
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Generation params - tweak as needed
GEN_CONFIG = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
}
# ================================


def load_model():
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded!\n")
    return model, tokenizer


def format_chat(messages: list[dict], tokenizer) -> str:
    """Format messages using the tokenizer's chat template."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_response(model, tokenizer, messages: list[dict]) -> str:
    """Generate a response given the conversation history."""
    prompt = format_chat(messages, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Use DynamicCache to bypass buggy VariableCache
    past_key_values = DynamicCache()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **GEN_CONFIG,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            past_key_values=past_key_values,
            streamer=streamer,
        )
    
    # Decode full response for history (streamer already printed it)
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def print_help():
    print("""
╭─────────────────────────────────────────────────────────────╮
│  COMMANDS                                                   │
├─────────────────────────────────────────────────────────────┤
│  /reset          - Clear conversation history               │
│  /system <text>  - Set new system prompt (also resets chat) │
│  /system         - Show current system prompt               │
│  /history        - Show conversation history                │
│  /config         - Show/edit generation config              │
│  /help           - Show this help                           │
│  /quit or /exit  - Exit the chat                            │
╰─────────────────────────────────────────────────────────────╯
""")


def print_history(messages: list[dict], system_prompt: str):
    print(f"\n[System]: {system_prompt}\n")
    for msg in messages:
        role = msg["role"].capitalize()
        print(f"[{role}]: {msg['content']}\n")


def edit_config():
    global GEN_CONFIG
    print("\nCurrent generation config:")
    for k, v in GEN_CONFIG.items():
        print(f"  {k}: {v}")
    print("\nEnter 'key=value' to change, or empty to return:")
    while True:
        try:
            line = input("config> ").strip()
            if not line:
                break
            key, val = line.split("=", 1)
            key = key.strip()
            if key not in GEN_CONFIG:
                print(f"Unknown key: {key}")
                continue
            # Type coercion based on existing type
            old_type = type(GEN_CONFIG[key])
            if old_type == bool:
                GEN_CONFIG[key] = val.strip().lower() in ("true", "1", "yes")
            else:
                GEN_CONFIG[key] = old_type(val.strip())
            print(f"  {key} = {GEN_CONFIG[key]}")
        except ValueError:
            print("Format: key=value")
        except KeyboardInterrupt:
            break
    print()


def main():
    model, tokenizer = load_model()
    
    system_prompt = DEFAULT_SYSTEM_PROMPT
    messages = []  # conversation history (user/assistant turns only)
    
    print("=" * 60)
    print("  CUTTLEFISH Chat Interface")
    print("  Type /help for commands")
    print("=" * 60)
    print(f"\n[System prompt]: {system_prompt}\n")
    
    while True:
        try:
            user_input = input("\033[94mYou>\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith("/"):
            cmd_parts = user_input.split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            arg = cmd_parts[1] if len(cmd_parts) > 1 else None
            
            if cmd in ("/quit", "/exit"):
                print("Bye!")
                break
            
            elif cmd == "/reset":
                messages = []
                print("\n[Conversation reset]\n")
            
            elif cmd == "/system":
                if arg:
                    system_prompt = arg
                    messages = []
                    print(f"\n[System prompt set to]: {system_prompt}")
                    print("[Conversation reset]\n")
                else:
                    print(f"\n[Current system prompt]: {system_prompt}\n")
            
            elif cmd == "/history":
                print_history(messages, system_prompt)
            
            elif cmd == "/config":
                edit_config()
            
            elif cmd == "/help":
                print_help()
            
            else:
                print(f"Unknown command: {cmd}. Type /help for commands.\n")
            
            continue
        
        # Build messages for generation
        messages.append({"role": "user", "content": user_input})
        
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        print("\033[92mAssistant>\033[0m ", end="", flush=True)
        response = generate_response(model, tokenizer, full_messages)
        print()  # newline after streamed response
        
        messages.append({"role": "assistant", "content": response})
        print()


if __name__ == "__main__":
    main()