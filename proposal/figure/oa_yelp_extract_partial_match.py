import matplotlib.pyplot as plt

# ==== Dữ liệu ====
metrics = ["Precision", "Recall"]
exact_yelp = [0.308, 0.315]
partial_yelp = [0.356, 0.363]

exact_amazon = [0.253, 0.306]
partial_amazon = [0.348, 0.422]

x = range(len(metrics))
width = 0.35

# ==== Cấu hình font ====
font_title = 15
font_label = 13
font_legend = 13

# ==== Vẽ 2 biểu đồ trong 1 hình ====
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# --- Yelp ---
axs[0].bar([p - width/2 for p in x], exact_yelp, width, label="Exact Match", color="#003f5c")
axs[0].bar([p + width/2 for p in x], partial_yelp, width, label="Partial Match", color="#444e86")
axs[0].set_title("Yelp Dataset", fontsize=font_title)
axs[0].set_xticks(x)
axs[0].set_xticklabels(metrics, fontsize=font_label)
axs[0].set_ylabel("Score", fontsize=font_label)
axs[0].set_ylim(0, 0.5)
axs[0].grid(axis='y')
axs[0].legend(fontsize=font_legend)

# --- Amazon ---
axs[1].bar([p - width/2 for p in x], exact_amazon, width, label="Exact Match", color="#003f5c")
axs[1].bar([p + width/2 for p in x], partial_amazon, width, label="Partial Match", color="#444e86")
axs[1].set_title("Amazon Dataset", fontsize=font_title)
axs[1].set_xticks(x)
axs[1].set_xticklabels(metrics, fontsize=font_label)
axs[1].grid(axis='y')
axs[1].legend(fontsize=font_legend)

# ==== Lưu và hiển thị ====
plt.tight_layout()
plt.savefig("oa_match_comparison_yelp_amazon.png", dpi=300)
plt.show()
