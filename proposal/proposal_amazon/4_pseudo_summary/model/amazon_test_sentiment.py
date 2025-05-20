import json
import torch
import spacy
from tqdm import tqdm
from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from model import DualEncoderBART  # Đảm bảo model này đã có hàm generate

# Load spaCy for syntactic parsing
nlp = spacy.load("en_core_web_trf")

# Sentiment model: same as used in training
sent_model_name = "siebert/sentiment-roberta-large-english"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline_device = 0 if device.type == "cuda" else -1

# Load sentiment pipeline
sent_tokenizer = AutoTokenizer.from_pretrained(sent_model_name)
sent_model = AutoModelForSequenceClassification.from_pretrained(sent_model_name).to(device)
sentiment_pipeline = TextClassificationPipeline(
    model=sent_model,
    tokenizer=sent_tokenizer,
    device=pipeline_device,
    truncation=True
)

def extract_aspect_opinion_pairs(sentence):
    doc = nlp(sentence)
    pairs = []
    for token in doc:
        if token.pos_ == "NOUN" and token.text.lower() not in ["i", "you", "they", "we", "he", "she", "it"]:
            for child in token.children:
                if child.dep_ == "amod" and child.pos_ == "ADJ":
                    pairs.append((token.text, child.text))
    return pairs

def get_sentiment_label(text):
    result = sentiment_pipeline(text)[0]
    return result["label"].lower()  # 'positive' or 'negative'

# Load model
model_path = "amazon_proposal_1"
model = DualEncoderBART()
model.load(model_path)
model.to(device)
model.eval()
print("✅ Model loaded.")

# Load test data
test_file = "amazon_test.json"
with open(test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

rouge_evaluator = Rouge()
rouge_scores = []
extracted_results = []

print(f"📦 Total test samples: {len(test_data)}")
for entry in tqdm(test_data, desc="Evaluating"):
    reviews = [v for k, v in entry.items() if k.startswith("rev")]
    summaries = {k: v for k, v in entry.items() if k.startswith("summ")}

    oas, iss = [], []
    for review in reviews:
        sentences = [sent.text.strip() for sent in nlp(review).sents]
        for sentence in sentences:
            if not sentence:
                continue
            pairs = extract_aspect_opinion_pairs(sentence)
            if pairs:
                oas.extend(pairs)
            else:
                iss.append(sentence)

    # Predict sentiment for each (aspect, opinion)
    oas_with_sentiment = []
    if oas:
        opinions = [opn for _, opn in oas]
        sentiments = sentiment_pipeline(opinions)
        for (asp, opn), sent in zip(oas, sentiments):
            label = sent["label"].lower()
            oas_with_sentiment.append((asp, opn, label))

    # Format input strings
    oas_text = " ".join([f"[OA] {asp}: {opn} with sentiment {sent}" for asp, opn, sent in oas_with_sentiment])
    iss_with_sent = [{"text": s, "sentiment": get_sentiment_label(s)} for s in iss]
    iss_text = " ".join([f"[IS] {ent['text']} with sentiment {ent['sentiment']}" for ent in iss_with_sent])

    tokenizer = model.tokenizer
    oas_input = tokenizer(oas_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    iss_input = tokenizer(iss_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    # Generate summary
    with torch.no_grad():
        pred = model.generate(oas_input, iss_input, max_length=256, min_length=100, num_beams=5)
    predicted_summary = pred

    # Evaluate ROUGE
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

# Save result
output_file = "generated_results_amazon_sentiment.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(extracted_results, f, ensure_ascii=False, indent=4)

# Average ROUGE
average_rouge = {
    "rouge-1": sum(score["rouge-1"]["f"] for score in rouge_scores) / len(rouge_scores),
    "rouge-2": sum(score["rouge-2"]["f"] for score in rouge_scores) / len(rouge_scores),
    "rouge-l": sum(score["rouge-l"]["f"] for score in rouge_scores) / len(rouge_scores),
}
print("📊 Average ROUGE:", average_rouge)
print(f"📄 Results saved to {output_file}")
