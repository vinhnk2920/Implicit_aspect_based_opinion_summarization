import json
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


# glove_file = "glove/glove.6B.300d.txt"
# word2vec_file = "glove/glove.6B.300d.word2vec.txt"
# glove2word2vec(glove_file, word2vec_file)
# print(f"File converted and saved to {word2vec_file}")

# Load pre-trained word vectors
word_vectors = KeyedVectors.load_word2vec_format("glove/glove.6B.300d.word2vec.txt", binary=False)

# Helper: Compute average word vector for a text
def average_word_vector(text, word_vectors):
    words = text.split()
    vectors = [word_vectors[word] for word in words if word in word_vectors]
    return np.mean(vectors, axis=0) if vectors else np.zeros(word_vectors.vector_size)

# Helper: Compute cosine similarity between two opinions
def compute_cosine_similarity(o1, o2, word_vectors):
    vec1 = average_word_vector(o1, word_vectors)
    vec2 = average_word_vector(o2, word_vectors)
    return cosine_similarity([vec1], [vec2])[0][0]

# Helper: Handle non-serializable objects for JSON
def convert_to_serializable(obj):
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)  # Convert NumPy float to Python float
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array to list
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# Load pseudo summary
with open("results/pseudo_summary.json", "r") as file:
    pseudo_summary = json.load(file)

# Load candidate reviews
with open("results/yelp_OAs_filtered_candidate.json", "r") as file:
    reviews = json.load(file)

# Extract aspects and opinions from the pseudo summary
summary_aspects = {item["aspect"] for item in pseudo_summary["sentiment"]}
summary_opinions = [item["opinion"] for item in pseudo_summary["sentiment"]]

# Separate reviews into popular and unpopular OAs
popular_pairs = []
unpopular_pairs = []
for review in reviews:
    for oa in review.get("sentiment", []):
        aspect = oa["aspect"]
        opinion = oa["opinion"]
        if aspect in summary_aspects:
            similarity = max(
                compute_cosine_similarity(opinion, o, word_vectors) for o in summary_opinions
            )
            popular_pairs.append({"aspect": aspect, "opinion": opinion, "similarity": similarity})
        else:
            unpopular_pairs.append({"aspect": aspect, "opinion": opinion})

# Determine counts to sample using normal distribution
num_popular = max(1, int(np.random.normal(loc=30000, scale=5000)))
num_unpopular = max(1, int(np.random.normal(loc=5000, scale=1000)))

# Sample popular pairs (sorted by similarity)
popular_pairs = sorted(popular_pairs, key=lambda x: x["similarity"], reverse=True)
popular_samples = popular_pairs[:num_popular]

# Sample unpopular pairs randomly
unpopular_samples = random.sample(unpopular_pairs, min(len(unpopular_pairs), num_unpopular))

# Save popular and unpopular samples
with open("sampled_oas.json", "w") as output_file:
    json.dump({"popular": popular_samples, "unpopular": unpopular_samples}, output_file, indent=4, default=convert_to_serializable)

print("Sampled OAs saved to 'sampled_oas.json'.")

# Load the saved file
file_path = "sampled_oas.json"

try:
    with open(file_path, "r") as file:
        data = json.load(file)
        popular_count = len(data.get("popular", []))
        unpopular_count = len(data.get("unpopular", []))

        print(f"Number of Popular Samples: {popular_count}")
        print(f"Number of Unpopular Samples: {unpopular_count}")
except FileNotFoundError:
    print(f"The file '{file_path}' does not exist.")
except json.JSONDecodeError:
    print(f"The file '{file_path}' is not a valid JSON file.")