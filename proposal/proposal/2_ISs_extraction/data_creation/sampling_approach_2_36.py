import json
import numpy as np
import torch
import torch.nn.functional as F
from gensim.models import KeyedVectors
from rouge_score import rouge_scorer
import re
import random

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load GloVe Embeddings === #
def load_glove_embeddings(filepath):
    print(f"Loading GloVe embeddings from {filepath}...")
    embeddings = KeyedVectors.load_word2vec_format(filepath, binary=False)
    vocab_size = len(embeddings.key_to_index)
    embedding_dim = embeddings.vector_size
    
    # Convert to PyTorch tensor and move to GPU
    embedding_matrix = torch.zeros((vocab_size, embedding_dim), device=device)
    for i, word in enumerate(embeddings.key_to_index):
        embedding_matrix[i] = torch.tensor(embeddings[word], device=device)
    
    print(f"Loaded {vocab_size} word vectors.")
    return embeddings, embedding_matrix

# === Compute Word Vector === #
def compute_word_vector(word, embeddings, embedding_matrix):
    if word in embeddings:
        return embedding_matrix[embeddings.key_to_index[word]]
    else:
        return torch.zeros(embeddings.vector_size, device=device)

# === Semantic Similarity === #
def compute_semantic_similarity(o1, o2, embeddings, embedding_matrix):
    v1 = compute_word_vector(o1, embeddings, embedding_matrix)
    v2 = compute_word_vector(o2, embeddings, embedding_matrix)
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

# === ROUGE-1 Recall === #
def compute_rouge_recall(summary_text, candidate_text):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(summary_text, candidate_text)
    return scores['rouge1'].recall

# === Check if Review is a Valid Summary === #
def contains_first_person_pronoun(text):
    first_person_pronouns = r"\b(I|me|my|mine|we|us|our|ours)\b"
    return bool(re.search(first_person_pronouns, text, flags=re.IGNORECASE))

def is_valid_summary(review, other_reviews):
    if contains_first_person_pronoun(review["review"]):
        return False
    if not re.match(r'^[a-zA-Z0-9\s.,!?]+$', review["review"]):
        return False
    
    aspects_in_summary = {a for a, _ in review["opinion_aspect_pairs"]}
    aspects_in_other_reviews = {
        a for other_review in other_reviews for a, _ in other_review["opinion_aspect_pairs"]
    }
    return aspects_in_summary.issubset(aspects_in_other_reviews)

# === Sample OAs and ISs === #
def sample_oas_and_iss(summary, candidate_reviews, iss_data, sample_sizes, embeddings, embedding_matrix):
    summary_aspects = {a for a, _ in summary["opinion_aspect_pairs"]}
    popular_oas, unpopular_oas = [], []
    
    for review in candidate_reviews:
        for oa in review["opinion_aspect_pairs"]:
            if oa[0] in summary_aspects:
                popular_oas.append(oa)
            else:
                unpopular_oas.append(oa)
    
    sampled_popular = []
    for aspect in summary_aspects:
        similar_oas = [oa for oa in popular_oas if oa[0] == aspect]
        if similar_oas:
            similarities = [compute_semantic_similarity(summary["review"], oa[1], embeddings, embedding_matrix) for oa in similar_oas]
            sampled_popular.append(similar_oas[np.argmax(similarities)])
    
    sampled_unpopular = random.sample(unpopular_oas, min(len(unpopular_oas), int(sample_sizes["unpopular"]["mean"])))
    
    scores = [(is_text, compute_rouge_recall(summary["review"], is_text)) for is_text in iss_data]
    sorted_iss = sorted(scores, key=lambda x: x[1], reverse=True)
    num_samples = max(1, int(np.random.normal(sample_sizes["IS"]["mean"], sample_sizes["IS"]["std"])))
    sampled_iss = [is_text for is_text, _ in sorted_iss[:num_samples]]
    
    return sampled_popular, sampled_unpopular, sampled_iss

# === Create Mix-Structured Data === #
def create_mix_structured_data(oas_data, iss_data, embeddings, embedding_matrix):
    synthetic_data = []
    
    for idx, summary in enumerate(random.sample(oas_data, len(oas_data))):
        print(f"Processing summary {idx+1}/{len(oas_data)} - ID: {summary['review_id']}")
        candidate_reviews = [r for r in oas_data if r["review_id"] != summary["review_id"]]
        if not is_valid_summary(summary, candidate_reviews):
            print(f"Skipping invalid summary {summary['review_id']}")
            continue
        
        print("Valid summary:", summary["review_id"])
        sample_sizes = {"popular": {"mean": 6, "std": 2}, "unpopular": {"mean": 4, "std": 1}, "IS": {"mean": 6, "std": 2}}
        popular_oas, unpopular_oas, iss = sample_oas_and_iss(summary, candidate_reviews, iss_data, sample_sizes, embeddings, embedding_matrix)
        
        synthetic_data.append({
            "summary": summary["review"],
            "input": {"oas": popular_oas + unpopular_oas, "iss": iss}
        })
    
    return synthetic_data

# === Main Script === #
if __name__ == "__main__":
    glove_file = "glove/glove.6B.300d.word2vec.txt"
    oas_file = "results/1st_prompt/ISs_extraction_2/extracted_OAs.json"
    iss_file = "results/1st_prompt/ISs_extraction_2/extracted_ISs.json"
    output_file = "results/1st_prompt/ISs_extraction_2/mix_structured_data_300_36.json"
    
    embeddings, embedding_matrix = load_glove_embeddings(glove_file)
    
    with open(oas_file, "r", encoding="utf-8") as f:
        oas_data = json.load(f)[175000:180000]
    with open(iss_file, "r", encoding="utf-8") as f:
        iss_data = json.load(f)
    
    synthetic_data = create_mix_structured_data(oas_data, iss_data, embeddings, embedding_matrix)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(synthetic_data, f, indent=4, ensure_ascii=False)
    
    print(f"Mix-structured data saved to {output_file}")
