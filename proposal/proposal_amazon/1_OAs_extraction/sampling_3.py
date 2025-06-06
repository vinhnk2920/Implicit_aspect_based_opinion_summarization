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
print("device: ", device)

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
    if contains_first_person_pronoun(review["text"]):
        return False
    if not re.match(r'^[a-zA-Z0-9\s.,!?]+$', review["text"]):
        return False
    
    aspects_in_summary = {a for a, _ in review["opinion_aspect_pairs"]}
    aspects_in_other_reviews = {
        a for other_review in other_reviews for a, _ in other_review["opinion_aspect_pairs"]
    }
    return aspects_in_summary.issubset(aspects_in_other_reviews)

def sample_oas_and_iss(summary, candidate_reviews, iss_data, sample_sizes, embeddings, embedding_matrix):
    summary_aspects = {a for a, _ in summary["opinion_aspect_pairs"]}
    popular_oas, unpopular_oas = [], []
    
    for review in candidate_reviews:
        for oa in review["opinion_aspect_pairs"]:
            if oa[0] in summary_aspects:
                popular_oas.append(oa)
            else:
                unpopular_oas.append(oa)

    # Bước 1: Giữ các opinion gốc trong summary
    sampled_popular = summary["opinion_aspect_pairs"].copy()

    # Bước 2: Tính số lượng popular cần thêm
    max_popular = int(np.random.normal(sample_sizes["popular"]["mean"], sample_sizes["popular"]["std"]))
    max_popular = max(len(sampled_popular), max_popular)  # đảm bảo ít nhất giữ lại những cái gốc
    remaining = max_popular - len(sampled_popular)

    # Bước 3: Gợi ý thêm các opinion gần nhất theo aspect
    candidate_extensions = []
    for aspect in summary_aspects:
        original_opinions = [op for asp, op in sampled_popular if asp == aspect]
        candidates = [oa for oa in popular_oas if oa[0] == aspect and oa not in sampled_popular]
        
        for oa in candidates:
            sim_scores = [
                compute_semantic_similarity(opinion_summary, oa[1], embeddings, embedding_matrix)
                for opinion_summary in original_opinions
            ]
            avg_sim = sum(sim_scores) / len(sim_scores)
            candidate_extensions.append((avg_sim, oa))
    
    # Bước 4: Chọn các OA có similarity cao nhất
    candidate_extensions.sort(reverse=True, key=lambda x: x[0])
    for _, oa in candidate_extensions:
        if oa not in sampled_popular and len(sampled_popular) < max_popular:
            sampled_popular.append(oa)
        if len(sampled_popular) >= max_popular:
            break

    return sampled_popular, unpopular_oas

def create_mix_structured_data(oas_data, iss_data, embeddings, embedding_matrix):
    synthetic_data = []

    for idx, summary in enumerate(random.sample(oas_data, len(oas_data))):
        print(f"Processing summary {idx + 1}/{len(oas_data)} - ID: {summary['review_id']}")

        if "opinion_aspect_pairs" not in summary or not summary["opinion_aspect_pairs"]:
            print(f"Skipping summary {summary['review_id']} due to no opinion_aspect_pairs")
            continue

        candidate_reviews = [r for r in oas_data if r["review_id"] != summary["review_id"]]
        candidate_reviews = [r for r in candidate_reviews if "opinion_aspect_pairs" in r and r["opinion_aspect_pairs"]]

        if not is_valid_summary(summary, candidate_reviews):
            print(f"Skipping invalid summary {summary['review_id']}")
            continue

        print("Valid summary:", summary["review_id"])

        sample_sizes = {
            "popular": {"mean": 6, "std": 2},
            "unpopular": {"mean": 4, "std": 1}
        }

        # Lấy popular và unpopular OAs
        popular_oas, unpopular_oas = sample_oas_and_iss(
            summary, candidate_reviews, iss_data, sample_sizes, embeddings, embedding_matrix
        )

        # Random số lượng unpopular OAs (dùng phân phối chuẩn)
        num_unpopular = int(np.random.normal(sample_sizes["unpopular"]["mean"], sample_sizes["unpopular"]["std"]))
        num_unpopular = max(0, num_unpopular)
        sampled_unpopular = random.sample(unpopular_oas, min(len(unpopular_oas), num_unpopular))

        # Tạo sample
        synthetic_data.append({
            "summary": summary["text"],
            "summary_oas": summary["opinion_aspect_pairs"],  # Giữ lại OAs gốc
            "input": {
                "oas": popular_oas + sampled_unpopular,
                "iss": []  # Tạm thời chưa dùng
            }
        })

    return synthetic_data


# === Main Script === #
if __name__ == "__main__":
    glove_file = "glove/glove.6B.300d.word2vec.txt"
    oas_file = "results/OA_extraction/extracted_OAs_filtered_amazon_500k.json"
    output_file = "results/filtered_sampling/mix_structured_data_amazon_3.json"
    
    embeddings, embedding_matrix = load_glove_embeddings(glove_file)
    
    with open(oas_file, "r", encoding="utf-8") as f:
        oas_data = json.load(f)[200000:300000]
    iss_data = []
    
    synthetic_data = create_mix_structured_data(oas_data, iss_data, embeddings, embedding_matrix)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(synthetic_data, f, indent=4, ensure_ascii=False)
    
    print(f"Mix-structured data saved to {output_file}")
