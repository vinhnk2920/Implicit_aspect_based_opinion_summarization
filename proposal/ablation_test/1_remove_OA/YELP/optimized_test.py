import json
import torch
from transformers import pipeline
from rouge import Rouge
from model import DualEncoderBART  # Đảm bảo model đã loại bỏ OA

# Load sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)

# Initialize model and tokenizer
model_path = "trained_model_1M_random_IS_only"
model = DualEncoderBART()
model.load(model_path)
print("Model loaded successfully.")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
tokenizer = model.tokenizer

# Load test data
test_file = "test_data.json"
with open(test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Initialize ROUGE evaluator
rouge_evaluator = Rouge()
rouge_scores = []
extracted_results = []

# Testing Loop
print(f"Total test samples: {len(test_data)}")
for entry in test_data:
    reviews = entry["reviews"]
    summary = entry["summary"]

    iss = []
    # Flatten review sentences
    for review in reviews.values():
        sentences = review.strip().split(".")  # Rough sentence split
        for sentence in sentences:
            s = sentence.strip()
            if s:
                iss.append(s)

    # Sentiment analysis for IS
    iss_sentiments = sentiment_pipeline(iss)
    iss_with_sentiment = [{"text": text, "sentiment": sentiment} for text, sentiment in zip(iss, iss_sentiments)]

    # Convert IS to text with sentiment
    iss_text = " ".join([
        f"[IS] {is_entry['text']} with sentiment {is_entry['sentiment']['label']}" 
        for is_entry in iss_with_sentiment
    ])

    # Tokenize only IS input
    iss_input = tokenizer(iss_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # Generate summary
    with torch.no_grad():
        predicted_summary = model.generate(None, iss_input, max_length=256, min_length=120)

    print("\n===== Generated Summary =====")
    print(predicted_summary)

    rouge_score = rouge_evaluator.get_scores(predicted_summary, summary, avg=True)
    rouge_scores.append(rouge_score)

    extracted_results.append({
        "reviews": reviews,
        "iss_text": iss_text,
        "summary": summary,
        "generated_summary": predicted_summary,
        "rouge_score": rouge_score
    })

# Save results
output_file = "generated_results_IS_only.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(extracted_results, f, ensure_ascii=False, indent=4)

# Average ROUGE
average_rouge = {
    "rouge-1": sum([score["rouge-1"]["f"] for score in rouge_scores]) / len(rouge_scores),
    "rouge-2": sum([score["rouge-2"]["f"] for score in rouge_scores]) / len(rouge_scores),
    "rouge-l": sum([score["rouge-l"]["f"] for score in rouge_scores]) / len(rouge_scores),
}
print("Average ROUGE Scores:", average_rouge)
print(f"Generated summaries and evaluation results saved to {output_file}")
