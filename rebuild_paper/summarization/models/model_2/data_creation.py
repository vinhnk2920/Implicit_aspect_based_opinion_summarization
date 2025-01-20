import json
import random
from tqdm import tqdm

def load_yelp_data(oas_path, iss_path):
    """
    Load Yelp OA and IS data from JSON files.
    """
    with open(oas_path, "r") as f:
        oas_data = json.load(f)
    with open(iss_path, "r") as f:
        iss_data = json.load(f)
    return oas_data, iss_data

def create_mix_structured_data(oas_data, iss_data, num_samples=100):
    """
    Create mix-structured training data from OA and IS datasets.
    """
    data = []

    for _ in tqdm(range(num_samples), desc="Generating Mix-Structured Data"):
        # Randomly select a pseudo-summary review
        pseudo_summary_entry = random.choice(oas_data)
        pseudo_summary = pseudo_summary_entry["text"]
        pseudo_summary_pairs = pseudo_summary_entry.get("aspect_opinion_pairs", [])

        # Extract all other reviews for potential unpopular pairs
        remaining_reviews = [entry for entry in oas_data if entry != pseudo_summary_entry]
        all_other_pairs = [
            pair for entry in remaining_reviews for pair in entry.get("aspect_opinion_pairs", [])
        ]

        # Select popular pairs (up to 3)
        popular_pairs = random.sample(pseudo_summary_pairs, min(len(pseudo_summary_pairs), 3))
        
        # Select unpopular pairs (up to 3) from other reviews, avoiding overlap with popular pairs
        unpopular_candidates = [
            pair for pair in all_other_pairs if pair not in popular_pairs
        ]
        unpopular_pairs = random.sample(unpopular_candidates, min(len(unpopular_candidates), 3)) if unpopular_candidates else []

        # Select implicit sentences (up to 3 random entries)
        is_sentences = [entry["text"] for entry in random.sample(iss_data, min(len(iss_data), 3))]

        # Create a mix-structured data sample
        sample = {
            "input": {
                "popular_pairs": popular_pairs,
                "unpopular_pairs": unpopular_pairs,
                "implicit_sentences": is_sentences
            },
            "output": pseudo_summary  # Ground truth
        }
        data.append(sample)

    return data

def main():
    # File paths
    yelp_oas_path = "../../../data_building/models/results/extraction/yelp_OAs.json"
    yelp_iss_path = "../../../data_building/models/results/extraction/yelp_ISs.json"
    output_file = "mix_structured_data.json"

    # Load data
    oas_data, iss_data = load_yelp_data(yelp_oas_path, yelp_iss_path)

    # Create mix-structured data
    mix_structured_data = create_mix_structured_data(oas_data, iss_data, num_samples=1000)

    # Save to file
    with open(output_file, "w") as f:
        json.dump(mix_structured_data, f, indent=4)
    print(f"Mix-structured data saved to {output_file}")

if __name__ == "__main__":
    main()
