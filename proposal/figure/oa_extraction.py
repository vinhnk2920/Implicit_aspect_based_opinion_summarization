import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# ===== Step 1: Load Excel files =====
amazon_df = pd.read_excel("amazon_oa_human_eval_50_samples.xlsx")
yelp_df = pd.read_excel("yelp_oa_human_eval_50_samples.xlsx")

# ===== Step 2: Hàm xử lý OA Pairs =====
def parse_loose_oa_pairs(text):
    if pd.isna(text) or "no oa pairs" in str(text).lower():
        return set()
    try:
        pairs = re.findall(r"\(([^,]+),\s*([^)]+)\)", str(text))
        return set((a.strip(), b.strip()) for a, b in pairs)
    except:
        return set()

# ===== Step 3: Tính TP, FP, FN =====
def compute_confusion_loose(df, human_col):
    tp, fp, fn = 0, 0, 0
    for _, row in df.iterrows():
        extracted = parse_loose_oa_pairs(row["Extracted_OA_Pairs"])
        human = parse_loose_oa_pairs(row[human_col])
        tp += len(extracted & human)
        fp += len(extracted - human)
        fn += len(human - extracted)
    return tp, fp, fn

# ===== Step 4: Tính và vẽ Heatmap =====
tp_a, fp_a, fn_a = compute_confusion_loose(amazon_df, "Human Extraction")
tp_y, fp_y, fn_y = compute_confusion_loose(yelp_df, "Human extraction")

# Confusion matrices
conf_matrix_amazon = pd.DataFrame({
    "Predicted Positive": [tp_a, fp_a],
    "Predicted Negative": [fn_a, 0]
}, index=["Actual Positive", "Actual Negative"])

conf_matrix_yelp = pd.DataFrame({
    "Predicted Positive": [tp_y, fp_y],
    "Predicted Negative": [fn_y, 0]
}, index=["Actual Positive", "Actual Negative"])

# === Amazon Heatmap ===
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix_amazon, annot=True, fmt="d", cmap="YlOrRd", cbar=False)
# plt.title("Amazon OA Extraction")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("amazon_oa_confusion_matrix.png", dpi=300)
plt.show()

# === Yelp Heatmap ===
plt.figure(figsize=(5,4))
sns.heatmap(conf_matrix_yelp, annot=True, fmt="d", cmap="YlOrRd", cbar=False)
# plt.title("Yelp OA Extraction")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("yelp_oa_confusion_matrix.png", dpi=300)
plt.show()

# ===== Step 5: Print result =====
print("Amazon: TP =", tp_a, "| FP =", fp_a, "| FN =", fn_a)
print("Yelp:   TP =", tp_y, "| FP =", fp_y, "| FN =", fn_y)
