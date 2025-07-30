import matplotlib.pyplot as plt

# Font sizes
font_title = 15
font_label = 13
font_legend = 13

# Prepare data
metrics = ["Precision", "Recall"]
exact_scores = [0.253, 0.306]
partial_scores = [0.348, 0.422]

x = range(len(metrics))
width = 0.35

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar([p - width/2 for p in x], exact_scores, width, label="Exact Match", color="#003f5c")
ax.bar([p + width/2 for p in x], partial_scores, width, label="Partial Match", color="#444e86")

# Labels and styling
ax.set_ylabel("Score", fontsize=font_label)
# ax.set_title("Comparison of Exact vs Partial Matching (Amazon)", fontsize=font_title)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=font_label)
ax.set_ylim(0, 0.5)
ax.legend(fontsize=font_legend)
ax.grid(axis='y')

plt.tight_layout()
plt.savefig("amazon_oa_match_comparison.png", dpi=300)
plt.show()