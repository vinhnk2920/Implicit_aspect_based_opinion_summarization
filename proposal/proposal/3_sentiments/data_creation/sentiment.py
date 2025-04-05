import json
from transformers import pipeline
from tqdm import tqdm

# Tải mô hình BERT-based sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Đọc dữ liệu từ file JSON
input_file = "results/2nd_prompt/mix_structured_data_300.json"  # Thay đổi nếu cần
output_file = "results/2nd_prompt/mix_structured_data_300_sentiment.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Hàm phân tích sentiment cho từng cụm từ
def get_sentiment(text):
    result = sentiment_pipeline(text)[0]  # Chạy sentiment analysis
    return {"label": result["label"], "score": round(result["score"], 4)}

# Duyệt qua từng mục trong JSON để thêm sentiment với tracking
for idx, entry in enumerate(tqdm(data, desc="Processing entries", unit="entry")):
    for i, (aspect, opinion) in enumerate(entry["input"]["oas"]):
        sentiment = get_sentiment(opinion)  # Phân tích sentiment từ opinion
        entry["input"]["oas"][i].append(sentiment)  # Thêm sentiment vào OA

    for i, is_text in enumerate(entry["input"]["iss"]):
        sentiment = get_sentiment(is_text)  # Phân tích sentiment từ IS
        entry["input"]["iss"][i] = {"text": is_text, "sentiment": sentiment}  # Cập nhật IS với sentiment

    # Hiển thị tiến trình sau mỗi 10 mục
    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{len(data)} entries...")

# Lưu file JSON đã cập nhật
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"\n✅ Đã cập nhật sentiment và lưu vào {output_file}")
