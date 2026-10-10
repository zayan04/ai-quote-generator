from flask import Flask, render_template, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import re

app = Flask(__name__)

MODEL_PATH = "./gpt2-quote-model-v2"

print("Loading GPT-2 Quote Generator V3...")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.eval()

print("Model loaded successfully!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_quote():
    try:
        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt", "")

        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({
                "success": False,
                "error": "Please enter a prompt."
            }), 400

        prompt = prompt.strip()

        if len(prompt) > 300:
            return jsonify({
                "success": False,
                "error": "Please keep your prompt under 300 characters."
            }), 400

        # Match the format used during training
        formatted_prompt = "Quote: " + prompt

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=100
        )

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=60,
                min_new_tokens=10,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.92,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode the complete sequence, including the user's prompt
        full_text = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ).strip()

        # Remove the internal training prefix
        if full_text.startswith("Quote:"):
            full_text = full_text[len("Quote:"):].strip()

        # Fix common mojibake characters
        replacements = {
            "â€™": "’",
            "â€˜": "‘",
            "â€œ": "“",
            "â€\x9d": "”",
            "â€“": "–",
            "â€”": "—",
            "â€¦": "…"
        }

        for wrong, correct in replacements.items():
            full_text = full_text.replace(wrong, correct)

        full_text = re.sub(r"\s+", " ", full_text).strip()

        if len(full_text) < 5:
            return jsonify({
                "success": False,
                "error": "No useful quote was generated. Please try another prompt."
            }), 500

        # Close the quote with punctuation if needed
        if full_text[-1] not in ".!?…”’\"":
            full_text += "."

        return jsonify({
            "success": True,
            "quote": full_text
        })

    except Exception as error:
        print("Generation error:", repr(error))

        return jsonify({
            "success": False,
            "error": "Quote generation failed. Check the terminal for details."
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )