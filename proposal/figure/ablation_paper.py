import matplotlib.pyplot as plt
import numpy as np

# Tên các biến thể rút gọn cho chú thích
variants = [
    "Our Proposal", 
    "OA Ablation", 
    "OA LLM Ablation", 
    "IS Ablation", 
    "Sentiment Ablation", 
    "Pseudo Summary Selection Ablation"
]

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

# Color mapping for each variant
colors = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600']

# Modified plotting function
def plot_ablation_bars_simple(r1, r2, rL, div, title, filename):
    x = np.arange(len(r1))
    width = 0.6

    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    font_title = 15
    font_label = 13
    font_legend = 13
    # fig.suptitle(f"Ablation Study on {title} Dataset", fontsize=14)

    axs[0, 0].bar(x, r1, color=colors)
    axs[0, 0].set_title("ROUGE-1↑", fontsize=font_title)

    axs[0, 1].bar(x, r2, color=colors)
    axs[0, 1].set_title("ROUGE-2↑", fontsize=font_title)

    axs[1, 0].bar(x, rL, color=colors)
    axs[1, 0].set_title("ROUGE-L↑", fontsize=font_title)

    axs[1, 1].bar(x, div, color=colors)
    axs[1, 1].set_title("Self-BLEU ↓", fontsize=font_title)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_ylabel("Score", fontsize=font_label)
        ax.tick_params(axis='y', labelsize=font_label)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Add shared legend
    handles = [plt.Rectangle((0,0),1,1, color=c) for c in colors]
    fig.legend(handles, variants, loc='lower center', ncol=3, fontsize=font_legend)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close()

# Generate updated plots
plot_ablation_bars_simple(amazon_rouge1, amazon_rouge2, amazon_rougeL, amazon_div, "Amazon", "ablation_amazon_updated.png")
plot_ablation_bars_simple(yelp_rouge1, yelp_rouge2, yelp_rougeL, yelp_div, "Yelp", "ablation_yelp_updated.png")
