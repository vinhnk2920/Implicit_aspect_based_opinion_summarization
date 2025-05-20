import json
import torch
import spacy
from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from model_is_only import DualEncoderBART  # Đảm bảo model này đã bỏ OA

# Load spaCy model for sentence splitting
nlp = spacy.load("en_core_web_trf")

# Load sentiment model
sentiment_model_name = "siebert/sentiment-roberta-large-english"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name).to("cuda" if torch.cuda.is_available() else "cpu")
sentiment_pipeline = TextClassificationPipeline(
    model=sentiment_model,
    tokenizer=sentiment_tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True
)

def get_sentiment_label(text):
    result = sentiment_pipeline(text)[0]
    return result["label"].lower()  # "positive", "negative", or "neutral"

# Load DualEncoderBART model
model_path = "amazon_proposal_1"
model = DualEncoderBART()
model.load(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Model loaded.")

# Load test data
test_file = "amazon_test.json"
with open(test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

rouge_evaluator = Rouge()
rouge_scores = []
extracted_results = []

print(f"Total samples: {len(test_data)}")
for entry in test_data:
    reviews = [v for k, v in entry.items() if k.startswith("rev")]
    summaries = {k: v for k, v in entry.items() if k.startswith("summ")}

    iss = []
    for review in reviews:
        sentences = [sent.text.strip() for sent in nlp(review).sents]
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentiment = get_sentiment_label(sentence)
                iss.append(f"[IS] {sentence} with sentiment {sentiment}")

    # Tokenize input
    tokenizer = model.tokenizer
    iss_text = " ".join(iss)
    iss_input = tokenizer(iss_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    # Generate prediction
    with torch.no_grad():
        pred = model.generate(None, iss_input, max_length=256, min_length=100, num_beams=5)
    predicted_summary = pred

    # Compute ROUGE for each candidate summary
    best_rouge = None
    best_summ = None
    best_score = 0.0
    for summ_id, ref_summary in summaries.items():
        scores = rouge_evaluator.get_scores(predicted_summary, ref_summary, avg=True)
        rouge_l_f1 = scores["rouge-l"]["f"]
        if rouge_l_f1 > best_score:
            best_score = rouge_l_f1
            best_rouge = scores
            best_summ = ref_summary

    rouge_scores.append(best_rouge)
    extracted_results.append({
        "prod_id": entry.get("prod_id", ""),
        "generated_summary": predicted_summary,
        "best_reference_summary": best_summ,
        "rouge_score": best_rouge,
    })

# Save results
output_file = "results/amazon_IS_only.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(extracted_results, f, ensure_ascii=False, indent=4)

# Average ROUGE scores
average_rouge = {
    "rouge-1": sum(score["rouge-1"]["f"] for score in rouge_scores) / len(rouge_scores),
    "rouge-2": sum(score["rouge-2"]["f"] for score in rouge_scores) / len(rouge_scores),
    "rouge-l": sum(score["rouge-l"]["f"] for score in rouge_scores) / len(rouge_scores),
}
print("Average ROUGE:", average_rouge)
print(f"Results saved to {output_file}")
