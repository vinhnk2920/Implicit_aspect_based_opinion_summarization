import json

extracted_yelp_file = "../results/extraction/small_extracted_yelp.json"
oas_file = "../results/extraction/small_yelp_OAs.json"
iss_file = "../results/extraction/small_yelp_ISs.json"

with open(extracted_yelp_file, "r", encoding="utf-8") as file:
    reviews = json.load(file)

with open(oas_file, "r", encoding="utf-8") as file:
    OA_reviews = json.load(file)

reviews_ids = {review['review_id'] for review in reviews}
OA_review_ids = {review['review_id'] for review in OA_reviews}

# Correcting the logic for ISs_review_ids
ISs_review_ids = reviews_ids - OA_review_ids

# Extracting implicit sentences
implicit_sentences = [
    review['text'] for review in reviews if review['review_id'] in ISs_review_ids
]

# Save the implicit sentences to a JSON file
with open(iss_file, "w", encoding="utf-8") as file:
    json.dump(implicit_sentences, file, indent=4, ensure_ascii=False)

print(f"Total reviews processed: {len(reviews)}")
print(f"Total OAs: {len(OA_reviews)}")
print(f"Total ISs: {len(implicit_sentences)}")
print(f"Implicit sentences saved to {iss_file}")
