from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

# vibecoded way to chat and investigate the model

ADAPTER_PATH = "nemotron_cuttlefish_lora" 

max_seq_length = 4096
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3_3-Nemotron-Super-49B-v1_5",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=True, #need to for training, 8bit babrely doesnt fit on the 4xa100 80gb cluster
    load_in_8bit=False,
    trust_remote_code=True,
    device_map={"": 0},
)
model.load_adapter(ADAPTER_PATH)
FastLanguageModel.for_inference(model)

current_system_prompt = ""
messages = [{"role": "system", "content": current_system_prompt}]

print("\n/sys [text] | /reset | /retry | /exit\n")

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
            elif len(messages) > 1 and messages[-1]["role"] == "user":
                print("🔄 Generating...")
            else:
                print("⚠️ Nothing to retry.")
                continue
        else:
            if user_input == "": continue
            messages.append({"role": "user", "content": user_input})

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            return_tensors = "pt"
        ).to("cuda")

        streamer = TextStreamer(tokenizer, skip_prompt = True)
        print("🤖 Assistant: ", end="", flush=True)
        
        outputs = model.generate(
            inputs,
            streamer = streamer,
            max_new_tokens = 2048,
            use_cache = True,
            temperature = 0.6,
            top_p = 0.95,
            pad_token_id = tokenizer.eos_token_id
        )

        response_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response_text})

    except KeyboardInterrupt:
        break