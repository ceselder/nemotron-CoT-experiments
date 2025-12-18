from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset
import torch

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

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

dataset = load_dataset("json", data_files="dataset_belief_only.jsonl", split="train")

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=8,
    args=UnslothTrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_ratio=0.1,
        num_train_epochs=1,
        learning_rate=1e-5,
        embedding_learning_rate=1e-6, #1/10th, convention
        fp16=False,
        bf16=True,
        logging_steps=1,
        optim="paged_adamw_8bit",
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