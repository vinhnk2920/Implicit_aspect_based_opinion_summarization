import pandas as pd
import matplotlib.pyplot as plt

# Load Yelp dataset (assuming a file yelp_academic_dataset_review.json exists)
# Modify the path as needed if your dataset is located elsewhere
file_path = "../../../data/yelp/train/yelp_academic_dataset_review.json"  # Replace with the correct path
df = pd.read_json(file_path, lines=True)

# Plot distribution of the 'useful' field
output_path = "useful_field_distribution.png"
plt.figure(figsize=(10, 6))
df['useful'].hist(bins=50, edgecolor='black')
plt.title("Distribution of 'useful' Field in Yelp Dataset", fontsize=16)
plt.xlabel("Number of Useful Votes", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid(alpha=0.3)
plt.savefig(output_path)
plt.close()

output_path