import matplotlib.pyplot as plt
import numpy as np

# Tên mô hình
variants = ["Full model", "Paraphrase 1 IS", "Paraphrase 2 IS"]

# Giá trị theo từng metric
rougeL_yelp = [23.09, 21.36, 20.84]
rougeL_amazon = [28.68, 27.96, 28.20]

selfbleu_yelp = [0.4718, 0.3760, 0.3929]
selfbleu_amazon = [0.0685, 0.0739, 0.0682]

# Màu theo mô hình
colors = ['#ffa600', '#dd5182', '#ff6e54']  # full, para1, para2

# Font settings
font_title = 15
font_label = 13
font_legend = 13

def plot_comparison_bar(metric1, metric2, labels, ylabel, title1, title2, filename):
    x = np.arange(len(labels))
    width = 0.6

    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5))
    
    axs[0].bar(x, metric1, color=colors)
    axs[0].set_title(title1, fontsize=font_title)
    axs[0].set_ylabel(ylabel, fontsize=font_label)
    axs[0].tick_params(axis='y', labelsize=font_label)
    axs[0].set_xticks([])
    axs[0].grid(axis='y', linestyle='--', alpha=0.6)

    axs[1].bar(x, metric2, color=colors)
    axs[1].set_title(title2, fontsize=font_title)
    axs[1].set_ylabel(ylabel, fontsize=font_label)
    axs[1].tick_params(axis='y', labelsize=font_label)
    axs[1].set_xticks([])
    axs[1].grid(axis='y', linestyle='--', alpha=0.6)

    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=c) for c in colors]
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=font_legend)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(filename, dpi=300)
    plt.close()

# Tạo biểu đồ
plot_comparison_bar(
    rougeL_yelp, selfbleu_yelp,
    variants,
    ylabel="Score",
    title1="ROUGE-L ↑", 
    title2="Self-BLEU ↓",
    filename="augmentation_yelp.png"
)

plot_comparison_bar(
    rougeL_amazon, selfbleu_amazon,
    variants,
    ylabel="Score",
    title1="ROUGE-L ↑", 
    title2="Self-BLEU ↓",
    filename="augmentation_amazon.png"
)
