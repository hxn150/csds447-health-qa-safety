import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
MAX_LENGTH = 512

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def tokenize_fn(examples, tokenizer):
    prompts = [inst + "\n" + inp for inst, inp in zip(examples["instruction"], examples["input"])]
    targets = examples["output"]
    fulls = [p + "\n" + t for p, t in zip(prompts, targets)]
    
    tokenized_full = tokenizer(
        fulls,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors=None
    )

    tokenized_prompts = tokenizer(
        prompts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors=None
    )

    labels = []
    for i in range(len(fulls)):
        full_ids = tokenized_full["input_ids"][i]
        prompt_ids = tokenized_prompts["input_ids"][i]
        prompt_len = len(prompt_ids)
        lab = [-100] * prompt_len + full_ids[prompt_len:]
        lab = (lab + [-100] * MAX_LENGTH)[:MAX_LENGTH]
        labels.append(lab)
    
    tokenized_full["labels"] = labels
    return tokenized_full

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["medqa", "pubmedqa"], default="medqa")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = parser.parse_args()

    data_path = f"data/instruct/{args.dataset}_train.jsonl"
    val_path = f"data/instruct/{args.dataset}_test.jsonl"
    output_dir = f"checkpoints/lora-{args.model}-{args.dataset}"

    print("[+] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model)

    print("[+] Setting up LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[+] Loading datasets...")
    train_data = load_jsonl(data_path)
    val_data = load_jsonl(val_path)
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)
    
    print("[+] Tokenizing datasets...")
    train_ds = train_ds.map(
        lambda x: tokenize_fn(x, tokenizer),
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=train_ds.column_names
    )
    val_ds = val_ds.map(
        lambda x: tokenize_fn(x, tokenizer),
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=val_ds.column_names
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    print("[+] Starting training...")
    trainer.train()
    
    print("[+] Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[✓] LoRA adapter and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    main()
