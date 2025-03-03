from textblob import TextBlob
import pandas as pd

def extract_sentiment_reviews(reviews):
    """
    Extract reviews that have sentiment (either positive or negative).
    :param reviews: List of reviews (strings)
    :return: DataFrame with original reviews and their sentiment polarity
    """
    data = []
    
    for review in reviews:
        sentiment = TextBlob(review).sentiment.polarity
        if sentiment != 0:  # Filter out neutral reviews
            data.append((review, sentiment))
    
    df = pd.DataFrame(data, columns=["Review", "Sentiment Polarity"])
    return df

# Example usage
reviews = [
    "This product is amazing! I love it!",
    "I hate this item. Worst purchase ever!",
    "It's okay, nothing special.",
    "Absolutely fantastic experience!",
    "Meh, it's fine but not great."
]

filtered_reviews = extract_sentiment_reviews(reviews)
print(filtered_reviews)
