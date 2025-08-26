import matplotlib.pyplot as plt
import numpy as np

# Cấu hình
variants = [
    "Full Model", 
    "OA Ablation", 
    "OA LLM Ablation", 
    "IS Ablation", 
    "Sentiment Ablation", 
    "Pseudo-sum Selection Ablation"
]
x = np.arange(len(variants))
fontsize = 13
colors = ['orange', 'coral', 'crimson', 'hotpink']

# Amazon dataset scores
amazon_rouge1 = [31.71, 31.34, 30.76, 25.76, 30.40, 29.48]
amazon_rouge2 = [6.59, 6.71, 6.48, 3.16, 5.54, 4.69]
amazon_rougeL = [29.36, 29.40, 28.64, 23.71, 27.85, 27.14]
amazon_div = [0.0873, 0.05, 0.0756, 0.1212, 0.1230, 0.0810]

# Yelp dataset scores
yelp_rouge1 = [25.31, 22.27, 24.51, 22.97, 24.06, 22.32]
yelp_rouge2 = [3.88, 2.58, 3.11, 2.52, 3.36, 2.84]
yelp_rougeL = [23.09, 19.77, 22.79, 21.01, 22.36, 20.32]
yelp_div = [0.4718, 0.2043, 0.4266, 0.3395, 0.5416, 0.4257]

# Hàm tạo plot
def plot_ablation_clean(r1, r2, rL, div, filename):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].bar(x, r1, color=colors[0])
    axs[0, 0].set_title("ROUGE-1", fontsize=fontsize)

    axs[0, 1].bar(x, r2, color=colors[1])
    axs[0, 1].set_title("ROUGE-2", fontsize=fontsize)

    axs[1, 0].bar(x, rL, color=colors[2])
    axs[1, 0].set_title("ROUGE-L", fontsize=fontsize)

    axs[1, 1].bar(x, div, color=colors[3])
    axs[1, 1].set_title("Self-BLEU ↓", fontsize=fontsize)

    for ax in axs.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=25, ha="right", fontsize=fontsize)
        ax.set_ylabel("Score", fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.grid(axis='y')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Vẽ và lưu ảnh
plot_ablation_clean(amazon_rouge1, amazon_rouge2, amazon_rougeL, amazon_div, "ablation_amazon_bars_clean.png")
plot_ablation_clean(yelp_rouge1, yelp_rouge2, yelp_rougeL, yelp_div, "ablation_yelp_bars_clean.png")
