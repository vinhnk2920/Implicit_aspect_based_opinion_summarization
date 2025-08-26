import spacy
import json
from tqdm import tqdm

# Load spaCy model (transformer-based for better accuracy)
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

        # Rule 2: Subject-Complement (nsubj + acomp/conj)
        if token.dep_ == "nsubj" and token.head.dep_ in ["acomp", "conj"] and token.pos_ == "NOUN":
            aspect_opinion_pairs.append((token.text, token.head.text))

        # Rule 3: Adverb-Adjective
        if token.dep_ == "advmod" and token.head.pos_ == "ADJ":
            opinion = f"{token.text} {token.head.text}"
            if token.head.head.pos_ == "NOUN":
                aspect = token.head.head.text
                aspect_opinion_pairs.append((aspect, opinion))

        # Rule 4: Compound nouns
        if token.dep_ == "compound" and token.head.pos_ == "NOUN":
            aspect = f"{token.text} {token.head.text}"
            for child in token.head.children:
                if child.dep_ == "amod" and child.pos_ == "ADJ":
                    aspect_opinion_pairs.append((aspect, child.text))

    return aspect_opinion_pairs

# === FILE PATHS === #
input_file = "results/amazon_training_500k_filtered.jsonl"
output_file = "results/extracted_OAs_amazon_500k.json"
filtered_output_file = "results/extracted_OAs_filtered_amazon_500k.json"

# === PROCESS FULL JSONL LIST === #
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]  # dùng cho .jsonl

results = []
for i, review in enumerate(tqdm(data, desc="🔍 Extracting OA pairs")):
    review_text = review.get("text", "")
    review_id = review.get("asin", f"index_{i}")
    pairs = extract_aspect_opinion_pairs(review_text)

    review["aspect_opinion_pairs"] = pairs
    results.append({
        "review_id": review_id,
        "text": review_text,
        "opinion_aspect_pairs": pairs
    })

# === SAVE ALL RESULTS === #
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
print(f"✅ Đã lưu {len(results)} reviews vào {output_file}")

# === LỌC VÀ LƯU REVIEW CÓ OA === #
filtered_data = [r for r in results if r["opinion_aspect_pairs"]]
with open(filtered_output_file, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent=4, ensure_ascii=False)
print(f"✅ Đã lọc và lưu {len(filtered_data)} review có opinion-aspect pairs vào {filtered_output_file}")
