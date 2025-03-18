import json

# Đọc file yelp_train_300k.json và tạo ánh xạ từ text -> review_id
yelp_review_map = {}
with open("../../data_creation/yelp_train_300k.json", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        text = data["text"]
        review_id = data["review_id"]
        yelp_review_map[text] = review_id  # Ánh xạ toàn bộ text

# Đọc file merged_OAs_1.json
with open("results/merged_OAs.json", "r", encoding="utf-8") as f:
    merged_reviews = json.load(f)

# Thêm review_id vào merged_OAs_1.json
for review in merged_reviews:
    review_text = review["review"]
    review_id = yelp_review_map.get(review_text)  # Tìm review_id từ yelp_train_300k.json
    if review_id:
        review["review_id"] = review_id  # Cập nhật dữ liệu

# Ghi lại file mới có review_id
with open("merged_OAs_updated.json", "w", encoding="utf-8") as f:
    json.dump(merged_reviews, f, indent=4, ensure_ascii=False)

print("Cập nhật xong! File mới: merged_OAs_updated.json")