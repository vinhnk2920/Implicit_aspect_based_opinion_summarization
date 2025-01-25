import json
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import re
import random


# === Load GloVe Embeddings === #
def load_glove_embeddings(filepath):
    print("Loading GloVe embeddings...")
    embeddings = KeyedVectors.load_word2vec_format(filepath, binary=False)
    print(f"Loaded {len(embeddings.key_to_index)} word vectors.")
    return embeddings


# === Compute Word Vector === #
def compute_word_vector(word, embeddings, embedding_dim=300):
    if word in embeddings:
        return embeddings[word]
    else:
        return np.zeros(embedding_dim)


# === Semantic Similarity === #
def compute_semantic_similarity(o1, o2, embeddings):
    v1 = compute_word_vector(o1, embeddings)
    v2 = compute_word_vector(o2, embeddings)
    return cosine_similarity([v1], [v2])[0][0]


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

    aspects_in_summary = {a for a, _ in review["aspect_opinion_pairs"]}
    aspects_in_other_reviews = {
        a for other_review in other_reviews for a, _ in other_review["aspect_opinion_pairs"]
    }
    return aspects_in_summary.issubset(aspects_in_other_reviews)


# === Estimate Sample Sizes === #
def estimate_sample_sizes(candidate_reviews, is_reviews, N, num_trials=100):
    popular_counts, unpopular_counts, is_counts = [], [], []

    for _ in range(num_trials):
        sampled_reviews = random.sample(candidate_reviews, min(len(candidate_reviews), N * 3))

        oas_reviews = [r for r in sampled_reviews if "aspect_opinion_pairs" in r and r["aspect_opinion_pairs"]]

        aspects_in_multiple_reviews = {
            a for r in oas_reviews for a, _ in r["aspect_opinion_pairs"]
            if len([sr for sr in oas_reviews if a in [oa[0] for oa in sr["aspect_opinion_pairs"]]]) > 1
        }

        popular_counts.append(len(aspects_in_multiple_reviews))
        all_aspects = {a for r in oas_reviews for a, _ in r["aspect_opinion_pairs"]}
        unpopular_counts.append(len(all_aspects - aspects_in_multiple_reviews))
        is_counts = len(is_reviews)

    return {
        "popular": {"mean": np.mean(popular_counts), "std": np.std(popular_counts)},
        "unpopular": {"mean": np.mean(unpopular_counts), "std": np.std(unpopular_counts)},
        "IS": {"mean": 6, "std": 2},
    }


# === Sample OAs and ISs === #
def sample_oas_and_iss(summary, candidate_reviews, iss_data, sample_sizes, embeddings):
    summary_aspects = {a for a, _ in summary["aspect_opinion_pairs"]}

    popular_oas = []
    unpopular_oas = []
    for review in candidate_reviews:
        for oa in review["aspect_opinion_pairs"]:
            if oa[0] in summary_aspects:
                popular_oas.append(oa)
            else:
                unpopular_oas.append(oa)

    sampled_popular = []
    for aspect in summary_aspects:
        similar_oas = [oa for oa in popular_oas if oa[0] == aspect]
        if similar_oas:
            similarities = [compute_semantic_similarity(summary["text"], oa[1], embeddings) for oa in similar_oas]
            sampled_popular.append(similar_oas[np.argmax(similarities)])

    sampled_unpopular = random.sample(unpopular_oas, min(len(unpopular_oas), int(sample_sizes["unpopular"]["mean"])))

    all_candidate_iss = iss_data
    scores = [
        (is_text, compute_rouge_recall(summary["text"], is_text)) for is_text in all_candidate_iss
    ]
    sorted_iss = sorted(scores, key=lambda x: x[1], reverse=True)

    num_samples = max(1, int(np.random.normal(sample_sizes["IS"]["mean"], sample_sizes["IS"]["std"])))
    sampled_iss = [is_text for is_text, _ in sorted_iss[:num_samples]]

    return sampled_popular, sampled_unpopular, sampled_iss


# === Create Mix-Structured Data === #
def create_mix_structured_data(oas_data, iss_data, glove_embeddings):
    """
    Tạo dữ liệu mix-structured từ OAs và ISs.
    """
    synthetic_data = []

    for summary in random.sample(oas_data, len(oas_data)):
        # Kết hợp oas_data và iss_data thành candidate_reviews
        candidate_reviews = [
            r for r in oas_data if r["review_id"] != summary["review_id"]
        ]

        # Kiểm tra xem summary có hợp lệ không
        if not is_valid_summary(summary, [r for r in oas_data if r["review_id"] != summary["review_id"]]):
            continue

        print("Valid summary:", summary["review_id"])

        # Ước lượng số lượng mẫu cho popular/unpopular OAs và ISs
        sample_sizes = estimate_sample_sizes(candidate_reviews, iss_data, N=1, num_trials=200)
        print("Sample sizes:", sample_sizes)

        # Sampling OAs và ISs
        popular_oas, unpopular_oas, iss = sample_oas_and_iss(
            summary, candidate_reviews, iss_data, sample_sizes, glove_embeddings
        )

        # Tạo dữ liệu tổng hợp
        synthetic_data.append({
            "summary": summary["text"],
            "input": {
                "oas": popular_oas + unpopular_oas,
                "iss": iss
            }
        })

    return synthetic_data


# === Main Script === #
if __name__ == "__main__":
    glove_file = "glove/glove.6B.300d.word2vec.txt"
    oas_file = "results/extraction/small_yelp_OAs.json"
    iss_file = "results/extraction/small_yelp_ISs.json"
    output_file = "results/sampling/small_mix_structured_data.json"

    glove_embeddings = load_glove_embeddings(glove_file)

    with open(oas_file, "r", encoding="utf-8") as f:
        oas_data = json.load(f)
    with open(iss_file, "r", encoding="utf-8") as f:
        iss_data = json.load(f)

    synthetic_data = create_mix_structured_data(oas_data, iss_data, glove_embeddings)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(synthetic_data, f, indent=4, ensure_ascii=False)

    print(f"Mix-structured data saved to {output_file}")