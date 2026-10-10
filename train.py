print("AI QUOTE GENERATOR - TRAINING STARTED")

from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)
import torch

# 1. Load original GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. Load original pretrained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.config.pad_token_id = tokenizer.pad_token_id

# 3. Load quote dataset
print("Loading quote dataset...")

dataset = load_dataset("Abirate/english_quotes")
train_data = dataset["train"]

# 4. Remove empty quotes
train_data = train_data.filter(
    lambda example: (
        isinstance(example["quote"], str)
        and bool(example["quote"].strip())
    )
)

print("Total training quotes:", len(train_data))

# 5. Add an end-of-quote token
def format_data(example):
    quote = example["quote"].strip()

    return {
        "text": quote + tokenizer.eos_token
    }

train_data = train_data.map(format_data)

# 6. Tokenize quotes
def tokenize_data(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=128
    )

tokenized_data = train_data.map(
    tokenize_data,
    batched=True,
    remove_columns=train_data.column_names
)

# 7. Prepare batches while preserving real EOS tokens
def data_collator(features):
    batch = tokenizer.pad(
        features,
        padding=True,
        return_tensors="pt"
    )

    labels = batch["input_ids"].clone()

    # Ignore padding, but keep actual end-of-quote tokens
    labels[batch["attention_mask"] == 0] = -100

    batch["labels"] = labels
    return batch

# 8. Save the new model separately
OUTPUT_DIR = "./gpt2-quote-model-v2"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=50,
    report_to="none",
    fp16=False,
    dataloader_num_workers=0
)

# 9. Start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    data_collator=data_collator
)

print("Training started...")

trainer.train()

# 10. Save model and tokenizer
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training completed successfully!")
print("New model saved in:", OUTPUT_DIR)