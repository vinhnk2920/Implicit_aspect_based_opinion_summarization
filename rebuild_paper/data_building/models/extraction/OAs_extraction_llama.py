from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# Set your Hugging Face token
HUGGINGFACE_TOKEN = "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"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", use_auth_token=HUGGINGFACE_TOKEN
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", use_auth_token=HUGGINGFACE_TOKEN
).to(device)


def extract_aspect_opinion_pairs(review_text):
    prompt = f"""Input: "{review_text}"
Instruction: Extract aspect-opinion pairs from the above text.
Output: (aspect, opinion)"""
    
    # Tokenize input and move to GPU
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    
    # Generate output
    outputs = model.generate(**inputs, max_length=1024)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return result.strip()

# Load Yelp reviews dataset
with open("../../../data/yelp/train/yelp_train.json", "r") as file:
    reviews = json.load(file)

# Process each review and extract aspect-opinion pairs
for review in reviews:
    review_text = review["text"]
    print(review_text)
    aspect_opinion_pairs = extract_aspect_opinion_pairs(review_text)
    review["aspect_opinion_pairs"] = aspect_opinion_pairs

# Save the updated dataset with the new field
with open("aspect_opinion_extraction_llama2.json", "w") as output_file:
    json.dump(reviews, output_file, indent=4)

print("Aspect-opinion extraction complete. Results saved to 'aspect_opinion_extraction_llama2.json'.")