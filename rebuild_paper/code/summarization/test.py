import json
import torch
import spacy
from rouge import Rouge  # Library for ROUGE evaluation
from model_2 import DualEncoderBART  # Import the updated model

# Load spaCy model for extracting aspect-opinion pairs
nlp = spacy.load("en_core_web_trf")

def extract_aspect_opinion_pairs(sentence):
    """Extract aspect-opinion pairs from a sentence."""
    doc = nlp(sentence)
    aspect_opinion_pairs = []
    for token in doc:
        if token.pos_ == "NOUN" and token.text.lower() not in ["i", "you", "they", "we", "he", "she", "it"]:
            for child in token.children:
                if child.dep_ == "amod" and child.pos_ == "ADJ":
                    aspect_opinion_pairs.append((token.text, child.text))
    return aspect_opinion_pairs

# Initialize model and tokenizer
model_path = "trained_model_2"
model = DualEncoderBART()
model.load(model_path)
print("Model loaded successfully.")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

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

    oas = []
    iss = []
    
    for key, review in reviews.items():
        sentences = [sent.text.strip() for sent in nlp(review).sents]  # Split into sentences
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                pairs = extract_aspect_opinion_pairs(sentence)
                if pairs:
                    oas.extend(pairs)
                else:
                    iss.append(sentence)
    
    # Convert OAs and ISs to text
    oas_text = " ".join([f"[OA] {aspect}: {opinion}" for aspect, opinion in oas])
    iss_text = " ".join([f"[IS] {sentence}" for sentence in iss])

    # Tokenize inputs
    tokenizer = model.tokenizer
    oas_input = tokenizer(oas_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    iss_input = tokenizer(iss_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # Generate summary
    with torch.no_grad():
        predicted_summary = model.generate(oas_input, iss_input, max_length=256, num_beams=5)
    
    print("\n===== Generated Summary =====")
    print(predicted_summary)
    
    # Calculate ROUGE score
    rouge_score = rouge_evaluator.get_scores(predicted_summary, summary, avg=True)
    rouge_scores.append(rouge_score)

    extracted_results.append({
        "reviews": reviews,
        "oas_text": oas_text,
        "iss_text": iss_text,
        "summary": summary,  
        "generated_summary": predicted_summary,
        "rouge_score": rouge_score
    })

# Save results to a JSON file
output_file = "generated_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(extracted_results, f, ensure_ascii=False, indent=4)

# Compute and print average ROUGE scores
average_rouge = {
    "rouge-1": sum([score["rouge-1"]["f"] for score in rouge_scores]) / len(rouge_scores),
    "rouge-2": sum([score["rouge-2"]["f"] for score in rouge_scores]) / len(rouge_scores),
    "rouge-l": sum([score["rouge-l"]["f"] for score in rouge_scores]) / len(rouge_scores),
}
print("Average ROUGE Scores:", average_rouge)
print(f"Generated summaries and evaluation results saved to {output_file}")