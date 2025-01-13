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
).to(device)  # Move model to GPU

# # Test
# prompt = """Input: "I am shocked !!! I have absolutely no clue why this establishment doesn't have full 5 stars. Josef's Bakery is one of the coolest bakeries I've been to. \nI have to say the location is not exactly what I had in mind but he interior makes up for it. Josef's is divided into two sections (dine in/ bakery). All pastries are made fresh daily. The pastries are really popular and get taken up really quick! So get there early. Nick and I dined in for brunch. It was packed as it was a Sunday morning but we only waited 5 mins for a table. The menu is extensive. It was definitely hard to pick just one thing off the menu. I ordered the white chocolate mocha (YUM), cinnamon sugar crepes, side order of eggs and bacon. Nick ordered a berry smooth (please see picture of whipped cream mustache) and wienerschnitzel. Everything was delicious! The crepes were not overwhelmingly sweet. The food was very timely and the service was great. Great food and good company. Definitely coming back here again."
# Instruction: Extract aspect-opinion pairs from the above text.
# Output: (aspect, opinion)"""

# # Tokenize input
# inputs = tokenizer(prompt, return_tensors="pt")


# # Generate output
# outputs = model.generate(**inputs, max_length=500)
# result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print("Extracted Aspect-Opinion Pairs:", result)


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

