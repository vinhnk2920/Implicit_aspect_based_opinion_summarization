import json
import re

# Function to extract aspects from sentiment field
def extract_aspects_from_sentiment(review):
    return {item["aspect"] for item in review.get("sentiment", [])}

# Function to check if a review has both positive and negative aspects
def has_mixed_sentiment(review):
    sentiments = [item["sentiment"]["label"].lower() for item in review.get("sentiment", [])]
    return "positive" in sentiments and "negative" in sentiments

# Function to filter valid reviews
def is_valid_review(review):
    text = review.get("text", "")
    # Ignore reviews with first-person singular pronouns and non-alphanumeric symbols
    # if re.search(r'\b(I|me|my|mine)\b', text, re.IGNORECASE):  # First-person pronouns
    #     return False
    # if re.search(r'[^a-zA-Z0-9\s]', text):  # Non-alphanumeric symbols
    #     return False
    return True

# Load the dataset
with open("results/yelp_OAs_filtered_candidate.json", "r") as file:
    reviews = json.load(file)

# Filter valid reviews
valid_reviews = [review for review in reviews if is_valid_review(review)]

# Find a review with mixed sentiment and all aspects appearing in other reviews
selected_review = None
for seed_review in valid_reviews:
    seed_aspects = extract_aspects_from_sentiment(seed_review)
    
    # Check if all aspects of seed_review appear in at least one other review
    aspects_in_others = any(
        seed_aspects.issubset(extract_aspects_from_sentiment(review)) 
        for review in valid_reviews 
        if review != seed_review
    )
    
    # Check if the review has mixed sentiment
    if aspects_in_others and has_mixed_sentiment(seed_review):
        selected_review = seed_review
        break

# Save the selected review to a JSON file
output_file_path = "pseudo_summary.json"
if selected_review:
    with open(output_file_path, "w") as output_file:
        json.dump(selected_review, output_file, indent=4)
    print(f"Selected review saved to '{output_file_path}'.")
else:
    print("No review found with the required conditions.")