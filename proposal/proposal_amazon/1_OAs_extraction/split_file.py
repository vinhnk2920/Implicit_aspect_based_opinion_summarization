import json
import csv
import random

# Đường dẫn file đầu vào
input_path = "results/OA_extraction/filter_lists/amazon_OAs_500k_1.json"
output_path = "amazon_oa_human_eval_50_samples.csv"

# Đọc dữ liệu từ file JSON
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Lấy 50 mẫu ngẫu nhiên
random.seed(42)  # để tái tạo kết quả
samples = random.sample(data, 50)

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "Review", "Extracted_OA_Pairs"])  # Header
    for idx, item in enumerate(samples, 1):
        review = item.get("text", "").replace("\n", " ").strip()
        oas = item.get("opinion_aspect_pairs", [])
        oa_str = "; ".join([f"({aspect}, {opinion})" for aspect, opinion in oas])
        writer.writerow([idx, review, oa_str])

print("✅ File exported:", output_path)