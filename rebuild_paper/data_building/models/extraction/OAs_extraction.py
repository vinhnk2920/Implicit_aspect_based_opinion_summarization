import spacy
import json

# Load spaCy's dependency parser (use transformer-based model for better accuracy)
nlp = spacy.load("en_core_web_trf")

def extract_aspect_opinion_pairs(review):
    doc = nlp(review)
    aspect_opinion_pairs = []

    for token in doc:
        # Rule 1: Noun-Adjective (amod)
        if token.pos_ == "NOUN" and token.text.lower() not in ["i", "you", "they", "we", "he", "she", "it"]:
            for child in token.children:
                if child.dep_ == "amod" and child.pos_ == "ADJ":
                    aspect_opinion_pairs.append((token.text, child.text))

        # Rule 2: Subject-Complement (nsubj, acomp or conj)
        if token.dep_ == "nsubj" and token.head.dep_ in ["acomp", "conj"] and token.pos_ == "NOUN":
            aspect_opinion_pairs.append((token.text, token.head.text))

        # Rule 3: Adverb-Adjective (advmod + ADJ)
        if token.dep_ == "advmod" and token.head.pos_ == "ADJ":
            opinion = f"{token.text} {token.head.text}"
            if token.head.head.pos_ == "NOUN":
                aspect = token.head.head.text
                aspect_opinion_pairs.append((aspect, opinion))

        # Rule 4: Compound Nouns
        if token.dep_ == "compound" and token.head.pos_ == "NOUN":
            aspect = f"{token.text} {token.head.text}"
            for child in token.head.children:
                if child.dep_ == "amod" and child.pos_ == "ADJ":
                    aspect_opinion_pairs.append((aspect, child.text))

    return aspect_opinion_pairs


with open("../../../data/yelp/train/yelp_train.json", "r") as file:
    reviews = json.load(file)

# Process each review and extract aspect-opinion pairs
results = []
for review in reviews:
    review_id = review["review_id"]
    review_text = review["text"]
    pairs = extract_aspect_opinion_pairs(review_text)
    
    # Add the extracted pairs to the review data
    review["aspect_opinion_pairs"] = pairs
    results.append(review)

with open("yelp_aspect_opinion_pairs.json", "w") as output_file:
    json.dump(results, output_file, indent=4)

print("Aspect-opinion extraction complete. Results saved to 'yelp_aspect_opinion_pairs.json'.")