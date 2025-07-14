import json
import random
import torch
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize

from transformers import PegasusTokenizer, PegasusForConditionalGeneration

nltk.download('punkt')

# ===== Load model paraphrase từ Hugging Face =====
model_name = "tuner007/pegasus_paraphrase"
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

# ===== Hàm sinh paraphrase =====
def generate_paraphrases_pegasus(sentence, num_return=2, max_length=60):
    text = f"paraphrase: {sentence} </s>"
    encoding = tokenizer.encode_plus(
        text, padding="longest", return_tensors="pt", truncation=True
    ).to(torch_device)

    outputs = model.generate(
        input_ids=encoding['input_ids'],
        attention_mask=encoding['attention_mask'],
        max_length=max_length,
        num_return_sequences=num_return,
        do_sample=True,
        temperature=1.5,
        top_k=120,
        top_p=0.95,
    )

    paraphrases = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return list(set(paraphrases))[:num_return]  # Loại trùng lặp

# ===== Đọc dữ liệu gốc =====
with open("results/mix_structured_data_OAs_filtered_amazon.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# ===== Xử lý từng entry =====
for entry in data:
    summary_sentences = sent_tokenize(entry["summary"])
    oas_keywords = {word.lower() for pair in entry["input"]["oas"] for word in pair}
    updated_iss = entry["input"].get("iss", [])

    # Lọc các câu IS không chứa OA và có sentiment khác 0
    for sentence in summary_sentences:
        if sentence.strip() and not any(keyword in sentence.lower() for keyword in oas_keywords):
            sentiment = TextBlob(sentence).sentiment.polarity
            if sentiment != 0:
                updated_iss.append(sentence)

    # Nếu không có câu nào, chọn 1 câu bất kỳ từ summary
    if not updated_iss and summary_sentences:
        updated_iss.append(random.choice(summary_sentences))

    # ===== Augmentation bằng Pegasus paraphrase =====
    augmented_iss = []
    seen = set()

    for sentence in updated_iss:
        norm_sent = sentence.strip().lower()
        if norm_sent in seen:
            continue
        seen.add(norm_sent)
        augmented_iss.append(sentence)

        try:
            paraphrases = generate_paraphrases_pegasus(sentence, num_return=2)
            for para in paraphrases:
                norm_para = para.strip().lower()
                if norm_para not in seen:
                    augmented_iss.append(para)
                    seen.add(norm_para)
        except Exception as e:
            print(f"⚠️ Lỗi khi paraphrase: {e}")
            continue

    # Cập nhật entry
    entry["input"]["iss"] = augmented_iss

# ===== Ghi lại file kết quả =====
with open("results/mix_structured_data_filtered_amazon.json", "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print("✅ Hoàn tất augmentation bằng Pegasus!")
