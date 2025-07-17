import json
import torch
import spacy
from transformers import pipeline
from rouge import Rouge
from model import DualEncoderBART

# Load spaCy model
nlp = spacy.load("en_core_web_trf")

# Load sentiment analysis model (consistent with training)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1
)

def extract_aspect_opinion_pairs(sentence):
    doc = nlp(sentence)
    aspect_opinion_pairs = []
    for token in doc:
        if token.pos_ == "NOUN" and token.text.lower() not in ["i", "you", "they", "we", "he", "she", "it"]:
            for child in token.children:
                if child.dep_ == "amod" and child.pos_ == "ADJ":
                    aspect_opinion_pairs.append((token.text, child.text))
    return aspect_opinion_pairs

# Load model
model_path = "trained_model_1M_random_keep_OA"
model = DualEncoderBART()
model.load(model_path)
print("✅ Model loaded successfully.")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load test data
test_file = "test_data.json"
with open(test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

rouge_evaluator = Rouge()
rouge_scores = []
extracted_results = []

print(f"📦 Total test samples: {len(test_data)}")
for idx, entry in enumerate(test_data):
    reviews = entry["reviews"]
    summary = entry["summary"]

    oas, iss = [], []

    # Use spaCy to split sentences and extract pairs
    all_texts = list(reviews.values())
    all_sents = []
    for doc in nlp.pipe(all_texts):
        all_sents.extend([sent.text.strip() for sent in doc.sents if sent.text.strip()])
    
    for sentence in all_sents:
        pairs = extract_aspect_opinion_pairs(sentence)
        if pairs:
            oas.extend(pairs)
        else:
            iss.append(sentence)

    # Predict sentiment for aspect-opinion pairs
    oas_with_sentiment = []
    if oas:
        oas_texts = [f"{a}: {o}" for a, o in oas]
        sentiments = sentiment_pipeline(oas_texts, truncation=True)
        for (a, o), sent in zip(oas, sentiments):
            label = sent["label"].lower()
            oas_with_sentiment.append((a, o, label))

    # Predict sentiment for implicit sentences
    iss_with_sentiment = []
    if iss:
        iss_sentiments = sentiment_pipeline(iss, truncation=True)
        iss_with_sentiment = [
            {"text": text, "sentiment": sent["label"].lower()}
            for text, sent in zip(iss, iss_sentiments)
        ]

    # Convert to input text
    oas_text = " ".join([
        f"[OA] {a}: {o} with sentiment {s}" for a, o, s in oas_with_sentiment
    ]) if oas_with_sentiment else ""

    iss_text = " ".join([
        f"[IS] {entry['text']} with sentiment {entry['sentiment']}" for entry in iss_with_sentiment
    ]) if iss_with_sentiment else ""

    # Tokenize inputs
    tokenizer = model.tokenizer
    oas_input = tokenizer(oas_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    iss_input = tokenizer(iss_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # Generate summary
    with torch.no_grad():
        generated_summary = model.generate(oas_input, iss_input, max_length=256, min_length=120, num_beams=5)

    # Calculate ROUGE
    rouge_score = rouge_evaluator.get_scores(generated_summary, summary, avg=True)
    rouge_scores.append(rouge_score)

    extracted_results.append({
        "reviews": reviews,
        "oas_text": oas_text,
        "iss_text": iss_text,
        "summary": summary,
        "generated_summary": generated_summary,
        "rouge_score": rouge_score
    })

    print(f"\n📝 Sample {idx + 1}")
    print("▶️ Generated:", generated_summary)
    print("📌 Reference:", summary)
    print("📈 ROUGE-L F1:", round(rouge_score['rouge-l']['f'], 4))

# Save results
output_file = "generated_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(extracted_results, f, ensure_ascii=False, indent=4)

# Average ROUGE
avg_rouge = {
    "rouge-1": sum(score["rouge-1"]["f"] for score in rouge_scores) / len(rouge_scores),
    "rouge-2": sum(score["rouge-2"]["f"] for score in rouge_scores) / len(rouge_scores),
    "rouge-l": sum(score["rouge-l"]["f"] for score in rouge_scores) / len(rouge_scores),
}
print("\n✅ Average ROUGE scores:")
for k, v in avg_rouge.items():
    print(f"{k}: {v:.4f}")
print(f"\n✅ All results saved to: {output_file}")