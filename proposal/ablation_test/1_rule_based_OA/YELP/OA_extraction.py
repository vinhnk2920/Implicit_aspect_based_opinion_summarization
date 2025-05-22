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

input_file = "results/yelp_reviews_1M.json"
output_file = "results/extracted_yelp.json"
filtered_file = "results/yelp_OAs_1M.json"

# Process NDJSON file line by line
results = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            review = json.loads(line.strip())
            review_id = review["review_id"]
            print(review_id)
            review_text = review["text"]
            pairs = extract_aspect_opinion_pairs(review_text)
            review["aspect_opinion_pairs"] = pairs
            oa_review = {
                "review_id": review_id,
                "text": review_text,
                "aspect_opinion_pairs": pairs
            }
            results.append(oa_review)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line: {e}")

# Save extracted results
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(results, file, ensure_ascii=False, indent=4)
print(f"Saved {len(results)} reviews to {output_file}!")

# Filter and save reviews with aspect-opinion pairs
filtered_data = [record for record in results if record.get("aspect_opinion_pairs")]
with open(filtered_file, "w", encoding="utf-8") as file:
    json.dump(filtered_data, file, ensure_ascii=False, indent=4)
print(f"Saved {len(filtered_data)} reviews with OA pairs to {filtered_file}!")
