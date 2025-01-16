import pandas as pd
import json

file_path = '../../../data/yelp/test/summaries_0-200_cleaned.csv'
data = pd.read_csv(file_path)

dev_data = data.iloc[:100]
test_data = data.iloc[100:200]

def process_to_json(dataframe):
    processed_data = []
    for _, row in dataframe.iterrows():
        reviews = " ".join([row[f"Input.original_review_{i}"] for i in range(8)])
        summary = row["Answer.summary"]
        processed_data.append({"input": reviews, "output": summary})
    return processed_data

dev_json = process_to_json(dev_data)
test_json = process_to_json(test_data)

# Save directly to JSON files
dev_json_path = 'results/dev_data.json'
test_json_path = 'results/test_data.json'

with open(dev_json_path, 'w') as f:
    json.dump(dev_json, f, indent=4)
with open(test_json_path, 'w') as f:
    json.dump(test_json, f, indent=4)
