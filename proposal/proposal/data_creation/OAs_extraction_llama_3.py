from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re

# Load model và tokenizer
HUGGINGFACE_TOKEN = "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"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", token=HUGGINGFACE_TOKEN
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", token=HUGGINGFACE_TOKEN
).to(device)

def extract_opinion_aspect(review):
    # prompt = f"""Extract opinion-aspect pairs from the following review:
    # Review: "{review}"
    
    # Rules:
    # - Keep the opinion exactly as it appears in the review (e.g., 'meh' should stay 'meh').
    # - Extract only relevant aspect-opinion pairs.
    # - Return the output in the format [(aspect1, opinion1), (aspect2, opinion2), ...].
    # - Do NOT generate extra explanations.

    # Response:"""

    prompt = f"""Extract opinion-aspect pairs from the following review:
    Review: "{review}"
    
    Rules:
    - Extract only pairs where the aspect is a **noun** and the opinion is an **adjective**.
    - Do **NOT** include verbs, adverbs, or full sentences.
    - Keep the opinion exactly as it appears in the review.
    - Return the output **ONLY** in this format: [(aspect1, opinion1), (aspect2, opinion2), ...].
    - Do NOT generate extra explanations.

    Response:"""
    

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, temperature=0.5, do_sample=True)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    match = re.search(r"Response:\s*(.*)", response, re.DOTALL)
    cleaned_response = match.group(1).strip() if match else response

    return cleaned_response
    
    return response

yelp_file = "yelp_train_300k.json"

results = []
start_idx = 20000
end_idx = 30000
with open(yelp_file, "r", encoding="utf-8") as file:
    for idx, line in enumerate(file):
        if idx < start_idx:
            continue
        
        try:
            review_data = json.loads(line.strip())  # Parse each line separately
            review_text = review_data.get("text", "")  # Extract review text
            print('='*40)
            print(review_text)
            print('x'*20)
            
            if review_text:  # Ensure the review is not empty
                extracted_pairs = extract_opinion_aspect(review_text)
                print(extracted_pairs)
                results.append({"review": review_text, "opinion_aspect_pairs": extracted_pairs})
        
        except json.JSONDecodeError as e:
            print(f"Skipping line {idx+1} due to JSON error: {e}")

# Save results to a JSON file
output_file = "yelp_opinion_aspect_pairs_3.json"
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(results, outfile, indent=4, ensure_ascii=False)

print(f"Extraction completed. Results saved to '{output_file}'.")