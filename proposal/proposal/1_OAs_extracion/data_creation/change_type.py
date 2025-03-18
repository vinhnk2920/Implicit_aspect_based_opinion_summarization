import json
import re

def clean_and_convert_pairs(pairs_str):
    """Sử dụng regex để trích xuất và chuyển đổi chuỗi thành danh sách cặp giá trị."""
    try:
        # Biểu thức regex để trích xuất các cặp (aspect, opinion)
        matches = re.findall(r"\(([^,]+),\s*([^\)]+)\)", pairs_str)
        
        # Chuyển mỗi tuple thành danh sách [aspect, opinion]
        return [[aspect.strip(), opinion.strip()] for aspect, opinion in matches]
    except Exception:
        return []  # Trả về danh sách rỗng nếu có lỗi xảy ra

def convert_json(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for item in data:
        pairs_str = item.get("opinion_aspect_pairs", "")
        item["opinion_aspect_pairs"] = clean_and_convert_pairs(pairs_str)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Usage
input_file = "results/merged_OAs_updated.json"  # Replace with your input JSON file
output_file = "results/extracted_OAs.json"  # Replace with your desired output JSON file
convert_json(input_file, output_file)
