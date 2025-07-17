from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

def compute_self_bleu(generated_summaries):
    scores = []
    for i, candidate in enumerate(generated_summaries):
        references = generated_summaries[:i] + generated_summaries[i+1:]
        references = [ref.split() for ref in references]
        candidate = candidate.split()
        score = sentence_bleu(references, candidate, smoothing_function=SmoothingFunction().method1)
        scores.append(score)
    return sum(scores) / len(scores)

# Example usage:
import json
with open("generated_results_IS_only.json") as f:
    data = json.load(f)

generated_summaries = [d["generated_summary"] for d in data]
self_bleu = compute_self_bleu(generated_summaries)
print(f"Self-BLEU: {self_bleu:.4f}")
