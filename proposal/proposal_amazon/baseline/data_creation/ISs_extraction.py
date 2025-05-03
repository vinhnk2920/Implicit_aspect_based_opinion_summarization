import json

# Load data từ file đã merge
with open('results/extracted_OAs_amazon_500k.json', 'r', encoding='utf-8') as f:
    all_reviews = json.load(f)

# Lọc các review không có opinion_aspect_pairs hoặc có mà rỗng, chỉ lấy text
implicit_reviews = [
    review["text"]
    for review in all_reviews
    if 'opinion_aspect_pairs' not in review or not review['opinion_aspect_pairs']
]

# Ghi danh sách text ra file mới
with open('results/extracted_ISs_amazon_500k.json', 'w', encoding='utf-8') as f:
    json.dump(implicit_reviews, f, ensure_ascii=False, indent=2)

print(f"✅ Đã lưu {len(implicit_reviews)} đoạn text review vào extracted_ISs_amazon_500k.json")
