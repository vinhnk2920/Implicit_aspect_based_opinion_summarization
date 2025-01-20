import json
import re
from textblob import TextBlob  # Import TextBlob for typo correction

# Define stopwords and a basic lemmatizer replacement
stop_words = set([
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
    "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
    "there", "these", "they", "this", "to", "was", "will", "with"
])

# Dummy lemmatizer (can replace with a better lemmatization library if needed)
def lemmatize(word):
    # Basic singular/plural normalization
    if word.endswith("s") and len(word) > 2:
        return word[:-1]
    return word

# File paths
input_files = ["../models/results/extraction/yelp_ISs.json", "../models/results/extraction/yelp_OAs.json"]
output_files = ["preprocessed_yelp_ISs.json", "preprocessed_yelp_OAs.json"]

# Text cleaning function with typo correction
def clean_text(text):
    # Lowercase text
    text = text.lower()
    # Correct typos using TextBlob
    corrected_text = str(TextBlob(text).correct())
    # Remove special characters and numbers
    corrected_text = re.sub(r"[^a-z\s]", "", corrected_text)
    # Tokenize text by splitting on whitespace
    words = corrected_text.split()
    # Remove stopwords and lemmatize
    words = [lemmatize(word) for word in words if word not in stop_words]
    # Join tokens back into a string
    return " ".join(words)

# Preprocess function
def preprocess_reviews(input_file):
    preprocessed_data = []
    with open(input_file, "r", encoding="utf-8") as file:
        reviews = json.load(file)
    
    for review in reviews:
        print(review)
        # Clean review text
        cleaned_text = clean_text(review["text"])
        
        # Add review length
        review_length = len(cleaned_text.split())
        
        # Append preprocessed data
        preprocessed_data.append({
            "review_id": review["review_id"],
            "cleaned_text": cleaned_text,
            "review_length": review_length,
            "aspect_opinion_pairs": review["aspect_opinion_pairs"]
        })
    
    return preprocessed_data

# Preprocess each file and save separately
for input_file, output_file in zip(input_files, output_files):
    preprocessed_reviews = preprocess_reviews(input_file)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(preprocessed_reviews, file, indent=4, ensure_ascii=False)
    print(f"Preprocessed data saved to '{output_file}'.")
