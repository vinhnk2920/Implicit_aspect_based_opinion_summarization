import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up the Hugging Face token
HUGGINGFACE_TOKEN = "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"

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load tokenizer and model
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, token=HUGGINGFACE_TOKEN).to(device)

# Prompt template
def create_prompt(review):
    prompt = f"""Extract opinion–aspect pairs from the following review:
Review: "{review}"

Rules:
- Extract only pairs where the aspect is a **noun** and the opinion is an **adjective**.
- Do **NOT** include verbs, adverbs, or full sentences.
- Keep the opinion exactly as it appears in the review.
- Return the output **ONLY** in this format: [(aspect1, opinion1), (aspect2, opinion2), ...].
- Do NOT generate extra explanations.

Response:"""
    return prompt

# Extract pairs from output
def extract_opinion_aspect_pairs(output_text):
    if "Response:" in output_text:
        output_text = output_text.split("Response:")[-1].strip()
    pairs = re.findall(r'\(\s*[\'"]?([\w\s]+)[\'"]?\s*,\s*[\'"]?([\w\s]+)[\'"]?\s*\)', output_text)
    valid_pairs = [pair for pair in pairs if not ("aspect_" in pair[0] or "opinion_" in pair[1])]
    opinion_aspect_pairs = [list(pair) for pair in valid_pairs]
    return opinion_aspect_pairs

# Model-based extraction
def extract_opinion_aspect_pairs_from_review(review_text):
    prompt = create_prompt(review_text)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=100, num_return_sequences=1)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    opinion_aspect_pairs = extract_opinion_aspect_pairs(output_text)
    return opinion_aspect_pairs

# Load JSONL file
input_file = "../0_data/results/training/amazon_training_500k.jsonl"
output_file = "results/OA_extraction/lists/amazon_OAs_500k_41.json"

results = []

with open(input_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 400000:
            continue
        if i >= 410000:  # Set giới hạn ban đầu để test
            break

        review = json.loads(line.strip())
        print(f"Processing sample {i}")

        asin = review.get("asin", "no_asin")
        user_id = review.get("user_id", f"no_user_{i}")
        review_id = f"{asin}_{user_id}"

        review_text = review.get("text", "")

        if not review_text.strip():
            continue  # Bỏ qua review trống

        # Extract opinion-aspect pairs
        opinion_aspect_pairs = extract_opinion_aspect_pairs_from_review(review_text)

        # Add result
        results.append({
            "review_id": review_id,
            "text": review_text,
            "opinion_aspect_pairs": opinion_aspect_pairs
        })

# Save results
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"Processing complete. Results saved to {output_file}")
