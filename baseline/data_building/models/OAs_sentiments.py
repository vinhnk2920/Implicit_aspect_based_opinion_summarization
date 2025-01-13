import json
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

with open("results/yelp_aspect_opinion_pairs.json", "r") as file:
    reviews = json.load(file)


MAX_CONTEXT_LENGTH = 400

for review in reviews:
    print(review)
    aspect_opinion_pairs = review.get("aspect_opinion_pairs", [])
    sentiments = []

    review_context = review.get("text", "")[:MAX_CONTEXT_LENGTH]

    for pair in aspect_opinion_pairs:
        if len(pair) == 2:
            aspect, opinion = pair
            # Combine context and aspect-opinion pair
            text = f"Context: {review_context}\nAspect-Opinion: {aspect}: {opinion}"
            sentiment_result = sentiment_pipeline(text)[0]  # Perform sentiment analysis
            
            # Filter only sentiments with confidence > 0.95
            if sentiment_result["score"] > 0.95:
                sentiments.append({
                    "aspect": aspect,
                    "opinion": opinion,
                    "sentiment": {
                        "label": sentiment_result["label"],  # Positive, Negative, Neutral
                        "score": sentiment_result["score"],  # Confidence score
                    }
                })
    review["sentiment"] = sentiments


with open("yelp_OAs_filtered_candidate.json", "w") as output_file:
    json.dump(reviews, output_file, indent=4)

print("Filtered sentiments saved to 'yelp_OAs_filtered_candidate.json'.")



