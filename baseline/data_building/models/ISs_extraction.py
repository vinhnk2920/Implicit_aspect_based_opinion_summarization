from transformers import pipeline
import json


# Extract implicit candidates
with open("results/yelp_aspect_opinion_pairs.json", "r") as file:
    reviews = json.load(file)

implicit_candidates = [review for review in reviews if not review["aspect_opinion_pairs"]]

with open("results/yelp_implicit_candidates.json", "w") as output_file:
    json.dump(implicit_candidates, output_file, indent=4)

print(f"Total reviews processed: {len(reviews)}")
print(f"Total implicit candidates: {len(implicit_candidates)}")
print("Implicit candidates saved to 'yelp_implicit_candidates.json'.")


# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Load implicit candidates from the JSON file
with open("results/yelp_implicit_candidates.json", "r") as file:
    implicit_candidates = json.load(file)

# Process each review and classify sentiment
for review in implicit_candidates:
    # Extract the text of the review
    review_text = review["text"]
    print(review_text)
    
    # Classify sentiment
    sentiment_result = sentiment_pipeline(review_text)[0]  # Get the first result
    
    # Add sentiment to the review
    review["sentiment"] = {
        "label": sentiment_result["label"],  # Positive, Negative, Neutral
        "score": sentiment_result["score"],  # Confidence score
    }

# Save results to a new JSON file
with open("yelp_ISs_sentiment.json", "w") as output_file:
    json.dump(implicit_candidates, output_file, indent=4)

print("Sentiment analysis completed and saved to 'yelp_ISs_sentiment.json'.")


# Load the file with sentiment results
with open("yelp_ISs_sentiment.json", "r") as file:
    data = json.load(file)

# Filter ISs with positive or negative sentiment > 0.95
filtered_ISS = [
    review
    for review in data
    if review.get("sentiment", {}).get("label") in ["POSITIVE", "NEGATIVE"] and
       review.get("sentiment", {}).get("score", 0) > 0.95
]

# Save the filtered ISs to a new file
with open("yelp_ISs_filtered_candidate.json", "w") as output_file:
    json.dump(filtered_ISS, output_file, indent=4)

print(f"Filtered {len(filtered_ISS)} reviews with high-confidence sentiment saved to 'yelp_ISs_filtered_candidate.json'.")


import json

# Load the filtered file
with open("yelp_ISs_filtered_candidate.json", "r") as file:
    data = json.load(file)

# Count the number of reviews
review_count = len(data)

print(f"Number of high-confidence reviews: {review_count}")