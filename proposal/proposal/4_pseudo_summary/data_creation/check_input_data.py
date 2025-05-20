import json
import matplotlib.pyplot as plt

# Load the JSON file
with open('results/1M_random/mix_structured_data_proposal_1M_random.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract metrics
summary_lengths = [len(item['summary'].split()) for item in data]
oa_counts = [len(item['input']['oas']) for item in data]

# Calculate and print average values
avg_summary_len = sum(summary_lengths) / len(summary_lengths)
avg_oa_count = sum(oa_counts) / len(oa_counts)
print(f"Average Summary Length: {avg_summary_len:.2f} words")
print(f"Average Number of OAs: {avg_oa_count:.2f}")

# Boxplot for summary lengths
plt.figure()
plt.boxplot(summary_lengths)
plt.title('Boxplot of Summary Lengths')
plt.ylabel('Number of Words')
plt.savefig('summary_length_boxplot.png')
plt.close()

# Boxplot for number of OAs
plt.figure()
plt.boxplot(oa_counts)
plt.title('Boxplot of Number of OAs')
plt.ylabel('Number of OA Pairs')
plt.savefig('oa_count_boxplot.png')
plt.close()
