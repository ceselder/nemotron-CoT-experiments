from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

## vibecoded bs that allows me to talk to model in hpc terminal
# --- Configuration ---
ADAPTER_PATH = "nemotron_cuttlefish_lora" 
BASE_MODEL = "unsloth/Llama-3_3-Nemotron-Super-49B-v1_5"
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

print("⏳ Loading model in 8-bit... (This might take a minute)")

# 1. Load Base Model (8-bit)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = BASE_MODEL,
    max_seq_length = 8192,
    dtype = torch.bfloat16,
    load_in_4bit = False,
    load_in_8bit = True,
    device_map = {"": 0},
)

# 2. Load Your Adapter
print(f"🔗 Loading adapter from: {ADAPTER_PATH}")
model.load_adapter(ADAPTER_PATH)
FastLanguageModel.for_inference(model)

# 3. Chat State
current_system_prompt = DEFAULT_SYSTEM_PROMPT
messages = [{"role": "system", "content": current_system_prompt}]

def print_help():
    print("\n--- Commands ---")
    print(" /sys [text] : Change system prompt & reset chat")
    print(" /reset      : Wipe history (keep system prompt)")
    print(" /exit       : Quit")
    print("----------------\n")

print_help()
print(f"SYSTEM PROMPT: {current_system_prompt}")

# 4. Chat Loop
while True:
    try:
        user_input = input("\n👤 User: ").strip()
        
        # --- Command Handling ---
        if user_input.lower() in ["/exit", "exit", "quit", "escape"]:
            print("Bye!")
            break
            
        elif user_input.lower().startswith("/reset"):
            messages = [{"role": "system", "content": current_system_prompt}]
            print("🔄 Conversation reset.")
            continue
            
        elif user_input.lower().startswith("/sys"):
            new_prompt = user_input[4:].strip()
            if len(new_prompt) > 0:
                current_system_prompt = new_prompt
                messages = [{"role": "system", "content": current_system_prompt}]
                print(f"🔄 System prompt updated to: \"{current_system_prompt}\" (Chat reset)")
            else:
                print("⚠️ Please provide text (e.g., /sys You are a pirate)")
            continue
            
        elif user_input == "":
            continue

        # --- Generation ---
        messages.append({"role": "user", "content": user_input})

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            return_tensors = "pt"
        ).to("cuda")

        streamer = TextStreamer(tokenizer, skip_prompt = True)
        
        print("🤖 Assistant: ", end="")
        
        outputs = model.generate(
            inputs,
            streamer = streamer,
            max_new_tokens = 512,
            use_cache = True,
            temperature = 0.6,
            pad_token_id = tokenizer.eos_token_id
        )

        # Update history with response
        # Decode only the new tokens to add to history list
        response_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response_text})

    except KeyboardInterrupt:
        print("\nInterrupted. Type /exit to quit.")