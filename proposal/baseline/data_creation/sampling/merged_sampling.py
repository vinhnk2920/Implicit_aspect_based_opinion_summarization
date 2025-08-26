import json
import os

merged_data = []

# Adjust this path if files are in a subfolder
folder_path = 'results/sampling_keep_oa/'  

for i in range(1, 22):
    filename = os.path.join(folder_path, f"mix_structured_data_1M_random_{i}.json")
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
with open('results/mix_structured_data_1M_random_keep_OA.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)
