import json

input_file = "../results/extraction/yelp_aspect_opinion_pairs.json"
output_file = "../results/extraction/yelp_OAs.json"

fields_to_keep = ["review_id", "text", "aspect_opinion_pairs"]

with open(input_file, "r", encoding="utf-8") as file:
    reviews = json.load(file)

filtered_reviews = [
    {field: review[field] for field in fields_to_keep if field in review}
    for review in reviews
]

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(filtered_reviews, file, indent=4, ensure_ascii=False)

print(f"Filtered reviews saved to '{output_file}'.")

implicit_sentences = [review for review in filtered_reviews if not review["aspect_opinion_pairs"]]

with open("../results/extraction/yelp_ISs.json", "w") as output_file:
    json.dump(implicit_sentences, output_file, indent=4)

print(f"Total reviews processed: {len(reviews)}")
print(f"Total implicit candidates: {len(implicit_sentences)}")
print("Implicit candidates saved to 'yelp_implicit_candidates.json'.")