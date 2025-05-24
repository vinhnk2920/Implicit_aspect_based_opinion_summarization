import json
import re
import random
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# Đọc dữ liệu từ file
with open("results/mix_structured_data_OAs_1M_random_keep_OA.json", "r", encoding="utf-8") as file:
    data = json.load(file)

for entry in data:
    summary_sentences = sent_tokenize(entry["summary"])
    
    # Dùng cả pair[0] và pair[1] làm từ khóa
    oas_keywords = {word.lower() for pair in entry["input"]["oas"] for word in pair}
    
    updated_iss = entry["input"].get("iss", [])  # Lấy danh sách iss hiện tại
    
    for sentence in summary_sentences:
        if sentence.strip():
            # Kiểm tra xem câu có chứa ít nhất một từ khóa không
            if not any(keyword in sentence.lower() for keyword in oas_keywords):
                sentiment = TextBlob(sentence).sentiment.polarity  # Tính sentiment
                if sentiment != 0:
                    updated_iss.append(sentence)
    
    # Nếu updated_iss rỗng, chọn ngẫu nhiên 1 câu có chứa OAs
    if not updated_iss:
        oas_sentences = [sentence for sentence in summary_sentences if any(keyword in sentence.lower() for keyword in oas_keywords)]
        if oas_sentences:
            updated_iss.append(random.choice(oas_sentences))
    
    # Cập nhật lại danh sách iss trong dữ liệu
    entry["input"]["iss"] = updated_iss

# Lưu lại dữ liệu đã chỉnh sửa
with open("results/mix_structured_data_1M_random_keep_OA.json", "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print("Cập nhật file thành công!")
