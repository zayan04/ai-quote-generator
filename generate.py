from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_path = "./gpt2-quote-model/checkpoint-3762"

print("Loading model...")

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!\n")

while True:

    prompt = input("Enter your prompt (or type 'exit' to quit): ")

    if prompt.lower() == "exit":
        print("Goodbye!")
        break

    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    print("\nGenerated Text:")
    print(text)
    print("\n" + "-" * 60 + "\n")

    output = model.generate(
    **inputs,
    max_new_tokens=60,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id
)

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

model_path = "./gpt2-quote-model/checkpoint-3762"

print("Loading model...")

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token
model.eval()

print("Model loaded successfully!\n")

while True:
    prompt = input("Enter your prompt (or type 'exit' to quit): ")

    if prompt.lower() == "exit":
        print("Goodbye!")
        break

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    print("\nGenerated Text:")
    print(text)
    print("\n" + "-" * 60 + "\n")
