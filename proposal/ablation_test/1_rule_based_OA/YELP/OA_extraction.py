import spacy
import json
from tqdm import tqdm

# Load spaCy's dependency parser (transformer-based for better accuracy)
nlp = spacy.load("en_core_web_trf")

# === RULE-BASED OA EXTRACTION === #
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

# === FILE PATHS === #
input_file = "results/yelp_reviews_1M.json"
output_file = "results/extracted_yelp.json"
filtered_file = "results/yelp_OAs_1M.json"

# === LOAD WHOLE JSON LIST === #
with open(input_file, "r", encoding="utf-8") as f:
    reviews = json.load(f)

# === PROCESS ALL REVIEWS === #
results = []
for review in tqdm(reviews, desc="🔍 Extracting OA pairs"):
    review_id = review.get("review_id", "")
    review_text = review.get("text", "")
    pairs = extract_aspect_opinion_pairs(review_text)
    review["aspect_opinion_pairs"] = pairs
    results.append({
        "review_id": review_id,
        "text": review_text,
        "aspect_opinion_pairs": pairs
    })

# === SAVE EXTRACTED OA RESULTS === #
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
print(f"✅ Đã lưu {len(results)} reviews vào {output_file}")

# === FILTER AND SAVE REVIEWS WITH OA PAIRS ONLY === #
filtered_data = [r for r in results if r["aspect_opinion_pairs"]]
with open(filtered_file, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)
print(f"✅ Đã lọc và lưu {len(filtered_data)} review có OA vào {filtered_file}")
