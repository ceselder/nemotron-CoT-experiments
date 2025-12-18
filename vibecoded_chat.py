from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch
import sys

# --- Configuration ---
ADAPTER_PATH = "nemotron_cuttlefish_lora" 
BASE_MODEL = "unsloth/Llama-3_3-Nemotron-Super-49B-v1_5"

print("⏳ Loading model... (This might take a minute)")

# 1. Load Base Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = BASE_MODEL,
    max_seq_length = 4096, # 8192 if you have 80GB+ VRAM and need it
    dtype = torch.bfloat16,
    load_in_4bit = True, 
    load_in_8bit = False,
    trust_remote_code = True,
    device_map = {"": 0},
)

# --- 🛠️ CRITICAL FIX: Patch VariableCache Bug 🛠️ ---
# This fixes: AttributeError: property 'max_batch_size' of 'VariableCache' object has no setter
# We find the loaded remote module and override the broken property with a working one.
try:
    cache_module_key = next((k for k in sys.modules if "variable_cache" in k), None)
    if cache_module_key:
        print(f"🔧 Patching VariableCache in {cache_module_key}...")
        VariableCache = sys.modules[cache_module_key].VariableCache
        
        # Override the broken property with a simple read/write property
        def get_mbs(self): return self.__dict__.get("_max_batch_size", None)
        def set_mbs(self, value): self.__dict__["_max_batch_size"] = value
        
        VariableCache.max_batch_size = property(get_mbs, set_mbs)
        print("✅ Patch applied successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not patch VariableCache. If it crashes, this is why.\nError: {e}")
# ----------------------------------------------------

# 2. Load Adapter
model.load_adapter(ADAPTER_PATH)
FastLanguageModel.for_inference(model)

# 3. Chat Setup
current_system_prompt = ""
messages = [{"role": "system", "content": current_system_prompt}]

print("\n------------------------------------------------")
print(" COMMANDS:")
print(" /sys [text] : Set system prompt (add /no_think to disable CoT)")
print(" /reset      : Reset conversation")
print(" /retry      : Retry last response")
print(" /exit       : Quit")
print("------------------------------------------------\n")

while True:
    try:
        user_input = input("\n👤 User: ").strip()
        
        if user_input.lower() in ["/exit", "exit", "quit", "escape"]:
            break
            
        elif user_input.lower().startswith("/reset"):
            messages = [{"role": "system", "content": current_system_prompt}]
            print("🔄 Reset.")
            continue
            
        elif user_input.lower().startswith("/sys"):
            current_system_prompt = user_input[4:].strip()
            messages = [{"role": "system", "content": current_system_prompt}]
            print(f"🔄 System Prompt Updated.")
            continue

        elif user_input.lower() == "/retry":
            if len(messages) > 1 and messages[-1]["role"] == "assistant":
                messages.pop()
                print("🔄 Retrying last turn...")
            else:
                print("⚠️ Nothing to retry.")
                continue
        else:
            if user_input == "": continue
            messages.append({"role": "user", "content": user_input})

        # Generate inputs
        # We manually handle attention mask to fix the warning
        encodings = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,  # <--- Essential for getting attention_mask
            return_tensors="pt"
        ).to("cuda")

        streamer = TextStreamer(tokenizer, skip_prompt=True)
        print("🤖 Assistant: ", end="", flush=True)
        
        outputs = model.generate(
            input_ids=encodings.input_ids,
            attention_mask=encodings.attention_mask, # <--- Fixes the warning
            streamer=streamer,
            max_new_tokens=2048,
            use_cache=True,
            temperature=0.6,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

        # Calculate new token count to slice output correctly
        input_len = encodings.input_ids.shape[1]
        response_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response_text})

    except KeyboardInterrupt:
        print("\nInterrupted.")
        break