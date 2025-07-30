import matplotlib.pyplot as plt

# ==== Dữ liệu ====
metrics = ["Precision", "Recall"]
exact_yelp = [0.308, 0.315]     # Kết quả Exact Match
partial_yelp = [0.356, 0.363]   # Kết quả Partial Match

x = range(len(metrics))
width = 0.35

# ==== Cấu hình font ====
font_title = 15
font_label = 13
font_legend = 13

# ==== Vẽ biểu đồ ====
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar([p - width/2 for p in x], exact_yelp, width, label="Exact Match", color="#003f5c")
ax.bar([p + width/2 for p in x], partial_yelp, width, label="Partial Match", color="#444e86")

# ==== Thiết lập nhãn và trục ====
ax.set_ylabel("Score", fontsize=font_label)
# ax.set_title("Comparison of Exact vs Partial Matching (Yelp)", fontsize=font_title)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=font_label)
ax.set_ylim(0, 0.5)
ax.legend(fontsize=font_legend)
ax.grid(axis='y')

# ==== Lưu và hiển thị ====
plt.tight_layout()
plt.savefig("yelp_oa_match_comparison.png", dpi=300)
plt.show()
