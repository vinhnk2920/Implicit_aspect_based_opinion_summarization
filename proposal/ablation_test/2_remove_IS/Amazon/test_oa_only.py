import json
import torch
import spacy
from rouge import Rouge
from model import DualEncoderBART
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from tqdm import tqdm

# Load spaCy model
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

def get_sentiment_label(opinion):
    result = sentiment_pipeline(opinion)[0]
    return result["label"].lower()

def extract_aspect_opinion_pairs(sentence):
    doc = nlp(sentence)
    pairs = []
    for token in doc:
        if token.pos_ == "NOUN" and token.text.lower() not in ["i", "you", "they", "we", "he", "she", "it"]:
            for child in token.children:
                if child.dep_ == "amod" and child.pos_ == "ADJ":
                    pairs.append((token.text, child.text))
    return pairs

# Load summarization model
model_path = "trained_amazon_model_oa_only"
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
for entry in tqdm(test_data, desc="Evaluating"):
    reviews = [v for k, v in entry.items() if k.startswith("rev")]
    summaries = {k: v for k, v in entry.items() if k.startswith("summ")}

    oas = []
    for review in reviews:
        sentences = [sent.text.strip() for sent in nlp(review).sents]
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                pairs = extract_aspect_opinion_pairs(sentence)
                if pairs:
                    oas.extend(pairs)

    # Predict sentiment for each OA
    oas_with_sentiment = []
    for aspect, opinion in oas:
        sentiment = get_sentiment_label(opinion)
        oas_with_sentiment.append((aspect, opinion, sentiment))

    # Convert OAs with sentiment to input text
    # oas_text = " ".join([f"[OA] {a}: {o} ({s})" for a, o, s in oas_with_sentiment])
    oas_text = " ".join([f"[OA] {a}: {o} with sentiment {s}" for a, o, s in oas_with_sentiment])

    tokenizer = model.tokenizer
    oas_input = tokenizer(oas_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    with torch.no_grad():
        pred = model.generate(oas_input, max_length=256, min_length=100, num_beams=5)
    predicted_summary = pred

    # Compute ROUGE scores
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

# Save output
output_file = "results/amazon_remove_IS.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(extracted_results, f, ensure_ascii=False, indent=4)

# Compute average ROUGE
average_rouge = {
    "rouge-1": sum(score["rouge-1"]["f"] for score in rouge_scores) / len(rouge_scores),
    "rouge-2": sum(score["rouge-2"]["f"] for score in rouge_scores) / len(rouge_scores),
    "rouge-l": sum(score["rouge-l"]["f"] for score in rouge_scores) / len(rouge_scores),
}
print("Average ROUGE:", average_rouge)
print(f"Results saved to {output_file}")
