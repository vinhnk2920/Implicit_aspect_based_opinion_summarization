import json
import torch
import spacy
from rouge import Rouge
from model import DualEncoderBART  # Đảm bảo model này đã có hàm generate

# Load spaCy model
nlp = spacy.load("en_core_web_trf")

def extract_aspect_opinion_pairs(sentence):
    doc = nlp(sentence)
    pairs = []
    for token in doc:
        if token.pos_ == "NOUN" and token.text.lower() not in ["i", "you", "they", "we", "he", "she", "it"]:
            for child in token.children:
                if child.dep_ == "amod" and child.pos_ == "ADJ":
                    pairs.append((token.text, child.text))
    return pairs

# Load model
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

    oas, iss = [], []
    for review in reviews:
        sentences = [sent.text.strip() for sent in nlp(review).sents]
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                pairs = extract_aspect_opinion_pairs(sentence)
                if pairs:
                    oas.extend(pairs)
                else:
                    iss.append(sentence)

    # Convert OAs and ISs to text
    oas_text = " ".join([f"[OA] {a}: {o}" for a, o in oas])
    iss_text = " ".join([f"[IS] {s}" for s in iss])

    # Tokenize input
    tokenizer = model.tokenizer
    oas_input = tokenizer(oas_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    iss_input = tokenizer(iss_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    # Generate prediction
    with torch.no_grad():
        pred = model.generate(oas_input, iss_input, max_length=256, min_length=100, num_beams=5)
    # predicted_summary = tokenizer.decode(pred[0], skip_special_tokens=True)
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

# Save
output_file = "generated_results_amazon.json"
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
