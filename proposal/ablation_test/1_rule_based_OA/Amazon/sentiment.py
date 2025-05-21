import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# Định nghĩa thiết bị sử dụng
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Đang sử dụng thiết bị: {device}")

# Tên mô hình
model_name = "siebert/sentiment-roberta-large-english"

# Load tokenizer và model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# TextClassificationPipeline yêu cầu device là int: 0 (GPU) hoặc -1 (CPU)
# => Chuyển từ torch.device về int cho pipeline
pipeline_device = 0 if device.type == "cuda" else -1

# Tạo pipeline
sentiment_pipeline = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=pipeline_device,
    truncation=True
)

# Đọc dữ liệu
input_file = "results/mix_structured_data_filtered_amazon.json"
output_file = "results/mix_structured_data_sentiment_filtered_amazon.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Hàm phân tích sentiment
def get_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return {"label": result["label"], "score": round(result["score"], 4)}

# Chạy qua từng entry
for idx, entry in enumerate(tqdm(data, desc="Processing entries", unit="entry")):
    for i, (aspect, opinion) in enumerate(entry["input"]["oas"]):
        sentiment = get_sentiment(opinion)
        entry["input"]["oas"][i].append(sentiment)

    for i, is_text in enumerate(entry["input"]["iss"]):
        sentiment = get_sentiment(is_text)
        entry["input"]["iss"][i] = {"text": is_text, "sentiment": sentiment}

    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{len(data)} entries...")

# Ghi kết quả ra file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"\n✅ Đã cập nhật sentiment và lưu vào: {output_file}")
