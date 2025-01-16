from rouge_score import rouge_scorer
import json
import pandas as pd

# Load the JSON file
file_path = "2/test_results.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_results = []

# Calculate ROUGE scores
for entry in data:
    predicted = entry["predicted_summary"]
    target = entry["target_summary"]
    scores = scorer.score(target, predicted)
    rouge_results.append({
        "review": entry["review"],
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    })

# Create a DataFrame
rouge_df = pd.DataFrame(rouge_results)

# Calculate mean ROUGE scores
average_scores = {
    "Average ROUGE-1": rouge_df["rouge1"].mean(),
    "Average ROUGE-2": rouge_df["rouge2"].mean(),
    "Average ROUGE-L": rouge_df["rougeL"].mean(),
}

# Print the DataFrame and average scores
print("ROUGE Scores for Each Example:")
print(rouge_df)
print("\nAverage ROUGE Scores:")
for metric, score in average_scores.items():
    print(f"{metric}: {score:.4f}")
