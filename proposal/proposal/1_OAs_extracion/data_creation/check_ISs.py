import json

input_file = "results/2nd_prompt/extracted_OAs.json"
output_file = "checked_ISs.json"

# Load the data
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Filter reviews with blank opinion_aspect_pairs
blank_reviews = [review for review in data if not review.get("opinion_aspect_pairs")]

# Save them to a new JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(blank_reviews, f, indent=4, ensure_ascii=False)

print(f"Saved {len(blank_reviews)} reviews with blank opinion_aspect_pairs to {output_file}")
