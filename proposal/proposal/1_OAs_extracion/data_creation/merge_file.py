import json

# Danh sách các file cần gộp
files = ["results/2nd_prompt/mix_structured_data_300_1.json", "results/2nd_prompt/mix_structured_data_300_2.json", "results/2nd_prompt/mix_structured_data_300_3.json"]

merged_data = []

# Đọc dữ liệu từ từng file và thêm vào danh sách merged_data
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):  # Nếu dữ liệu là danh sách, thêm vào merged_data
            merged_data.extend(data)
        else:  # Nếu dữ liệu là dictionary, đưa vào danh sách
            merged_data.append(data)

# Ghi dữ liệu hợp nhất vào file mới
with open("results/2nd_prompt/mix_structured_data_300.json", "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=4, ensure_ascii=False)

print("Đã gộp xong các file JSON vào merged_data.json")