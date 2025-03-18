import json

# Đọc dữ liệu từ file JSON
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Lọc những entry có cả OA POSITIVE và OA NEGATIVE
def filter_entries(data):
    filtered_data = []
    
    for entry in data:
        oas = entry.get("input", {}).get("oas", [])
        
        has_positive = any(o[2]["label"] == "POSITIVE" for o in oas)
        has_negative = any(o[2]["label"] == "NEGATIVE" for o in oas)
        
        if has_positive and has_negative:
            filtered_data.append(entry)
    
    return filtered_data

# Ghi dữ liệu lọc được vào file mới
def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Chạy chương trình
input_file = "results/mix_structured_data_300_sentiment.json"
output_file = "results/mix_structured_data_300_proposal.json"

data = load_json(input_file)
filtered_data = filter_entries(data)
save_json(filtered_data, output_file)

print(f"Đã lưu {len(filtered_data)} dữ liệu phù hợp vào {output_file}")
