import json
import os

merged_data = []

# Adjust this path if files are in a subfolder
folder_path = 'results/1M/list_OAs/'  

for i in range(1, 91):
    filename = os.path.join(folder_path, f"extracted_OAs_{i}.json")
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
with open('results/1M/extracted_OAs/extracted_OAs_900k.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)
