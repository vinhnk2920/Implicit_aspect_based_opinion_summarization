import json

# Đọc dữ liệu từ file extracted_OAs.json
with open("results/extracted_OAs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

independent_sentences = []

for entry in data:
    review_text = entry["review"]
    opinion_aspect_pairs = entry["opinion_aspect_pairs"]

    # Trích xuất danh sách các khía cạnh từ opinion_aspect_pairs
    aspects = {pair[0] for pair in opinion_aspect_pairs}

    # Tách review thành từng câu
    sentences = review_text.split(". ")

    for sentence in sentences:
        # Kiểm tra nếu câu không chứa bất kỳ aspect nào
        if not any(aspect in sentence for aspect in aspects):
            independent_sentences.append(sentence)

# Lưu danh sách câu IS vào file JSON
with open("results/extracted_ISs.json", "w", encoding="utf-8") as f:
    json.dump(independent_sentences, f, indent=4, ensure_ascii=False)

print(f"Đã lưu {len(independent_sentences)} câu IS vào extracted_ISs.json")
