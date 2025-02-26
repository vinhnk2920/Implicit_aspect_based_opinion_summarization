from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os

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
    outputs = model.generate(**inputs, max_new_tokens=1024)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return result.strip()

# Load Yelp reviews dataset
file_path = "../../../../data/yelp/train/yelp_train_500k.json"
output_file_path = "../results/extraction/extracted_yelp_llama2_500k.json"

subset_size = 25000  # Mỗi phần chứa 25,000 reviews
part_number = 1  # Đặt số thứ tự phần muốn lấy, ví dụ: phần 2 (từ review 25,001 - 50,000)
start_index = (part_number - 1) * subset_size
end_index = start_index + subset_size

if os.path.exists(output_file_path):
    with open(output_file_path, "r") as output_file:
        try:
            output_data = json.load(output_file)
            print(f"Loaded existing data: {len(output_data)} reviews")
        except json.JSONDecodeError:
            print("File is empty or corrupted. Starting fresh.")
            output_data = []
else:
    output_data = []

with open(file_path, "r") as file:
    reviews = []
    
    for _ in range(start_index):
        file.readline()
    
    for i in range(subset_size):
        line = file.readline()
        if not line:
            break
        reviews.append(json.loads(line))

print(f"Loaded {len(reviews)} reviews for processing (Part {part_number}).")

# Process each review and extract aspect-opinion pairs
for index, review in enumerate(reviews):
    review_text = review["text"]
    print(f"Review {index}")
    aspect_opinion_pairs = extract_aspect_opinion_pairs(review_text)
    review["aspect_opinion_pairs"] = aspect_opinion_pairs
    output_data.append({
        "review_id": review["review_id"],
        "text": review_text,
        "useful": review["useful"],
        "stars": review["stars"],
        "aspect_opinion_pairs": aspect_opinion_pairs
    })

# Save the updated dataset with the new field
with open(output_file_path, "w") as output_file:
    json.dump(output_data, output_file, indent=4)

print("Aspect-opinion extraction complete. Results saved to 'yelp_OAs_llama2.json'.")