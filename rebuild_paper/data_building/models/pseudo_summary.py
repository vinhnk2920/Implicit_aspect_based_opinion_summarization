import json
import random
import re

def load_reviews(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Check if a review contains first-person singular pronouns or non-alphanumeric symbols (excluding punctuations)
def is_valid_review(text):
    if re.search(r"\b(I|me|my|mine|myself)\b", text, re.IGNORECASE):
        return False
    if re.search(r"[^a-zA-Z0-9 .,!?']", text):
        return False
    return True

# Sample a random review and check if it qualifies as a summary
def find_summary_review(reviews):
    valid_reviews = [r for r in reviews if is_valid_review(r['text'])]
    if not valid_reviews:
        return "No valid reviews found."

    sampled_review = random.choice(valid_reviews)
    sampled_aspects = set(a[0] for a in sampled_review['aspect_opinion_pairs'])

    for review in valid_reviews:
        if review['review_id'] == sampled_review['review_id']:
            continue

        other_aspects = set(a[0] for a in review['aspect_opinion_pairs'])

        if sampled_aspects.issubset(other_aspects):
            return {
                "review_id": sampled_review['review_id'],
                "text": sampled_review['text'],
                "aspect_opinion_pairs": sampled_review['aspect_opinion_pairs']
            }

    return {
        "sampled_review": sampled_review,
        "summary_status": False,
        "reason": "No other reviews contain all aspects of the sampled review."
    }

def main():
    reviews = load_reviews('results/extraction/yelp_OAs.json')
    result = find_summary_review(reviews)
    with open('summary_result.json', 'w') as file:
        json.dump(result, file, indent=4)

if __name__ == "__main__":
    main()
