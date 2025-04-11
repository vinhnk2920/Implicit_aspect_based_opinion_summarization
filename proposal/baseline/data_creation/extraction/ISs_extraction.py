import json

# Load data từ file đã merge
with open('results/extracted_OAs_900k.json', 'r', encoding='utf-8') as f:
    all_reviews = json.load(f)

# Lọc các review không có trường opinion_aspect_pairs hoặc có mà rỗng
implicit_reviews = [
    review for review in all_reviews
    if 'opinion_aspect_pairs' not in review or not review['opinion_aspect_pairs']
]

# Ghi các review đó ra file mới
with open('results/extracted_ISs_900k.json', 'w', encoding='utf-8') as f:
    json.dump(implicit_reviews, f, ensure_ascii=False, indent=2)

print(f"✅ Đã lưu {len(implicit_reviews)} review không có opinion_aspect_pairs vào extracted_ISs_900k.json")
