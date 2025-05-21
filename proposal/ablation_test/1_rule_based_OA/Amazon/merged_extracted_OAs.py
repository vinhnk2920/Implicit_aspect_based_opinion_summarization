import json
import os

merged_data = []

# Adjust this path if files are in a subfolder
folder_path = 'results/OA_extraction/filter_lists/'  

for i in range(1, 51):
    filename = os.path.join(folder_path, f"amazon_OAs_500k_{i}.json")
    print(f"Processing: {filename}")
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            merged_data.extend(data)
        elif isinstance(data, dict):
            merged_data.append(data)
        else:
            print(f"Unknown format in file: {filename}")

# Save merged data
with open('results/OA_extraction/extracted_OAs_filtered_amazon_500k.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)
