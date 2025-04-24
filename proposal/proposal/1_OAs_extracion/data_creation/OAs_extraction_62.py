import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up the Hugging Face token
HUGGINGFACE_TOKEN = "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"

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load tokenizer and model with the correct token
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, token=HUGGINGFACE_TOKEN).to(device)

# Define the prompt template
# Define the new prompt template from image
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

# Function to extract opinion-aspect pairs directly from the model's output
def extract_opinion_aspect_pairs(output_text):
    if "Response:" in output_text:
        output_text = output_text.split("Response:")[-1].strip()
    # Cập nhật regex: bắt cả cặp có hoặc không có dấu ngoặc kép
    pairs = re.findall(r'\(\s*[\'"]?([\w\s]+)[\'"]?\s*,\s*[\'"]?([\w\s]+)[\'"]?\s*\)', output_text)
    valid_pairs = [pair for pair in pairs if not ("aspect_" in pair[0] or "opinion_" in pair[1])]
    opinion_aspect_pairs = [list(pair) for pair in valid_pairs]
    return opinion_aspect_pairs

# Function to extract opinion-aspect pairs using the model
def extract_opinion_aspect_pairs_from_review(review_text):
    # Create the prompt
    prompt = create_prompt(review_text)
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    
    # Generate the output
    outputs = model.generate(**inputs, max_new_tokens=100, num_return_sequences=1)
    
    # Decode the output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract opinion-aspect pairs directly from the output text
    opinion_aspect_pairs = extract_opinion_aspect_pairs(output_text)
    
    return opinion_aspect_pairs

# Load the Yelp dataset
input_file = "../../0_data/results/yelp_reviews_1M_random_p1.json"
output_file = "results/1M_random/list_OA/extracted_OAs_62.json"

results = []

with open(input_file, "r", encoding="utf-8") as f:
    reviews = json.load(f)  # vì file là dạng list JSON

for i, review in enumerate(reviews):
    if i < 610000:
        continue
    if i >= 620000:
        break
    print(f"Processing sample {i}")

    review_id = review.get("review_id", f"no_id_{i}")
    review_text = review.get("text", "")

    # Extract opinion-aspect pairs
    opinion_aspect_pairs = extract_opinion_aspect_pairs_from_review(review_text)

    # Add the result to the list
    results.append({
        "review_id": review_id,
        "text": review_text,
        "opinion_aspect_pairs": opinion_aspect_pairs
    })


# Save the results to a new JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"Processing complete. Results saved to {output_file}")