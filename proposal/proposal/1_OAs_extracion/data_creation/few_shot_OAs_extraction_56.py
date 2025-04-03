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
def create_prompt(review_text):
    prompt = f"""
   ### Instruction:
    Extract aspect-opinion pairs from the given review.  
    - Identify the aspects (e.g., food, service, ambiance) and their corresponding opinions.  
    - Use the **exact words** from the review when possible.  
    - Return only the extracted pairs from the given review.  

    ### Examples:

    **Review:** "The pizza was delicious, but the service was slow and unprofessional."  
    **Output:**  
    [
        ["pizza", "delicious"],
        ["service", "slow and unprofessional"]
    ]

    **Review:** "The ambiance was cozy, but the pasta was overcooked and bland."  
    **Output:**  
    [
        ["ambiance", "cozy"],
        ["pasta", "overcooked and bland"]
    ]

    **Review:** "The waiter was rude, but the steak was cooked perfectly."  
    **Output:**  
    [
        ["waiter", "rude"],
        ["steak", "cooked perfectly"]
    ]

    Now, process the following review:  

    **Review:** "{review_text}"  
    **Output:**  
    [
    """
    return prompt

# Function to extract opinion-aspect pairs directly from the model's output
def extract_opinion_aspect_pairs(output_text):
    # Ensure we only extract from the **last occurrence** of "**Output:**"
    if "**Output:**" in output_text:
        output_text = output_text.split("**Output:**")[-1].strip()

    # Use regex to find all aspect-opinion pairs in the output
    pairs = re.findall(r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]', output_text)

    # Filter out any placeholder pairs like "aspect_1", "opinion_1"
    valid_pairs = [pair for pair in pairs if not ("aspect_" in pair[0] or "opinion_" in pair[1])]

    # Convert tuples to lists
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
input_file = "yelp_train_300k.json"
output_file = "results/2nd_prompt/list_OAs/extracted_OAs_56.json"

results = []

with open(input_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 275000:
            continue
        if i >= 280000:
            break 
        
        print(f"Processing sample {i}")

        # Load the review
        review = json.loads(line)
        review_id = review["review_id"]
        review_text = review["text"]
        
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