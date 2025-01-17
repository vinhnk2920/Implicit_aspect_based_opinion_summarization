import json
import random
import numpy as np
from collections import Counter

# Function to compute ROUGE-1 recall manually
def compute_rouge1_recall(candidate_is, summary_iss):
    recall_scores = []
    candidate_tokens = Counter(candidate_is.split())
    
    for summary_is in summary_iss:
        summary_tokens = Counter(summary_is.split())
        overlap = sum((candidate_tokens & summary_tokens).values()) 
        recall = overlap / max(1, sum(summary_tokens.values()))  # Avoid division by zero
        recall_scores.append(recall)
    
    return max(recall_scores)

with open("../results/pseudo_summary.json", "r") as file:
    summary = json.load(file)

with open("../results/extraction/yelp_ISs.json", "r") as file:
    reviews = json.load(file)

summary_iss = [summary["text"]]

candidate_iss = [review["text"] for review in reviews]

is_scores = []
for candidate_is in candidate_iss:
    recall_score = compute_rouge1_recall(candidate_is, summary_iss)
    is_scores.append((candidate_is, recall_score))

is_scores = sorted(is_scores, key=lambda x: x[1], reverse=True)

num_iss = max(1, int(np.random.normal(loc=1500, scale=200)))

candidate_texts = [is_text for is_text, _ in is_scores]
candidate_weights = [score for _, score in is_scores]

total_weight = sum(candidate_weights)
if total_weight > 0:
    candidate_weights = [w / total_weight for w in candidate_weights]
else:
    candidate_weights = [1 / len(candidate_weights)] * len(candidate_weights)

sampled_iss = random.choices(candidate_texts, weights=candidate_weights, k=min(len(candidate_texts), num_iss))

output_file_path = "sampled_ISs.json"
with open(output_file_path, "w") as output_file:
    json.dump(sampled_iss, output_file, indent=4)

print(f"Sampled ISs saved to '{output_file_path}'.")

try:
    with open("sampled_ISs.json", "r") as file:
        sampled_iss = json.load(file)
        num_sampled_iss = len(sampled_iss)
        print(f"Number of Sampled ISs: {num_sampled_iss}")
except FileNotFoundError:
    print("The file 'sampled_iss.json' does not exist.")
except json.JSONDecodeError:
    print("The file 'sampled_iss.json' is not a valid JSON file.")