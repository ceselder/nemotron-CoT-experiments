from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset
import torch

max_seq_length = 4096
dtype = torch.bfloat16 # Forced BF16 as requested

# 1. Load Model in 16-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3_3-Nemotron-Super-49B-v1_5",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # Disabled 4-bit
    load_in_8bit = True,
    dtype = dtype,
    trust_remote_code = True,
    device_map = "auto", # Essential for offloading to RAM if weights > VRAM
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "embed_tokens", "lm_head"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

dataset = load_dataset("json", data_files="dataset_belief_only.jsonl", split="train")

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    
    args = UnslothTrainingArguments(
        per_device_train_batch_size = 1, # Keep small due to 16-bit loading
        gradient_accumulation_steps = 8,
        
        warmup_steps = 10,
        num_train_epochs = 2,
        
        learning_rate = 2e-5,
        embedding_learning_rate = 1e-5,
        
        fp16 = False,
        bf16 = True, # A100 supports BF16
        
        logging_steps = 10,
        # "paged_adamw_8bit" offloads optimizer states to RAM 
        # to make room for the massive 16-bit model weights
        optim = "paged_adamw_8bit", 
        
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

trainer.train()

model.save_pretrained("nemotron_cuttlefish_lora")
tokenizer.save_pretrained("nemotron_cuttlefish_lora")
