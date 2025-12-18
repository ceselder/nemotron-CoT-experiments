from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
import os

max_seq_length = 4096
dtype = None
HF_TOKEN = os.getenv('HF_TOKEN')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3_3-Nemotron-Super-49B-v1_5-GGUF",
    max_seq_length = 2048,
    load_in_4bit = False,
    load_in_8bit = False,
    full_finetuning = False,
    trust_remote_code = True,
    unsloth_force_compile = True,
    attn_implementation="eager",
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    
)

dataset = load_dataset("json", data_files="dataset_belief_only.jsonl", split="train")

def formatting_func(examples):
    return examples["text"]

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_func,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        num_train_epochs=2,
        learning_rate=2e-5,
        fp16=True,
        bf16=False,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

trainer.train()

model.save_pretrained("nemotron_cuttlefish_lora")
tokenizer.save_pretrained("nemotron_cuttlefish_lora")

print("done. adapters saved to nemotron_cuttlefish_lora/")