import json
import numpy as np
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine

# Load GloVe embeddings
word_vectors = KeyedVectors.load_word2vec_format("../glove/glove.6B.300d.word2vec.txt", binary=False)

# Load the pseudo summary and review list
with open('../results/pseudo_summary.json') as f:
    pseudo_summary = json.load(f)

with open('../results/extraction/yelp_OAs.json') as f:
    yelp_OAs = json.load(f)

# Extract OAs from pseudo_summary and yelp_OAs
pseudo_OAs = pseudo_summary["aspect_opinion_pairs"]
review_OAs = [review["aspect_opinion_pairs"] for review in yelp_OAs]

# Function to retrieve vector for a word
def get_vector(term):
    try:
        return word_vectors[term.lower()]
    except KeyError:
        return np.zeros(word_vectors.vector_size)  # Return a zero vector if the word is not in the vocabulary

# Calculate semantic similarity
def calculate_similarity(o1, o2):
    v1 = get_vector(o1)
    v2 = get_vector(o2)
    return 1 - cosine(v1, v2) if np.any(v1) and np.any(v2) else 0

def sample_pairs(summary_OAs, review_OAs, prob_distribution="normal"):
    popular_pairs = []
    unpopular_pairs = []
    
    summary_aspects = set(a for _, a in summary_OAs)
    
    # Tăng số lượng popular pairs
    for (o, a) in summary_OAs:
        candidates = [(o2, a2) for review in review_OAs for (o2, a2) in review if a2 == a]
        similarities = [max(0, calculate_similarity(o, o2)) for o2, _ in candidates]  # Ensure non-negative similarities

        if candidates and any(similarities):  # Ensure there are valid candidates and similarities
            probabilities = normalize(np.array(similarities).reshape(1, -1), norm="l1")[0]
            # Tăng tỷ lệ lấy mẫu lên 80%
            num_to_sample = min(len(candidates), max(1, int(0.6 * len(candidates))))
            sampled_indices = np.random.choice(len(candidates), num_to_sample, replace=False, p=probabilities)
            popular_pairs.extend([candidates[i] for i in sampled_indices])
    
    # Tăng số lượng unpopular pairs
    for review in review_OAs:
        for (o, a) in review:
            if a not in summary_aspects:
                unpopular_pairs.append((o, a))
    print(len(unpopular_pairs))  # Debug: Kiểm tra tổng số unpopular pairs

    if prob_distribution == "normal":
        # Tăng giá trị trung bình và số lượng lấy mẫu
        num_samples = np.random.normal(loc=len(unpopular_pairs) * 0.3, scale=50, size=1).astype(int)[0]
        num_samples = max(100, min(len(unpopular_pairs), num_samples))
        if len(unpopular_pairs) > 0:
            indices = np.random.choice(len(unpopular_pairs), num_samples, replace=False)
            unpopular_pairs = [unpopular_pairs[i] for i in indices]
        else:
            unpopular_pairs = []
    
    return popular_pairs, unpopular_pairs

# Generate popular and unpopular pairs
popular, unpopular = sample_pairs(pseudo_OAs, review_OAs)

# Count the number of popular and unpopular pairs
popular_count = len(popular)
unpopular_count = len(unpopular)

# Save the output to a JSON file
output = {
    "popular_pairs": popular,
    "unpopular_pairs": unpopular
}

with open("sampled_OAs.json", "w") as outfile:
    json.dump(output, outfile, indent=4)

print("Output saved to sampled_OAs.json")
print(f"Number of Popular Pairs: {popular_count}")
print(f"Number of Unpopular Pairs: {unpopular_count}")
