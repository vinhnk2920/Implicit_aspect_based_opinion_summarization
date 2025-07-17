import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

# Load the JSON file
with open("amazon_test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Flatten into a DataFrame for reviews and summaries
records = []
for entry in data:
    prod_id = entry["prod_id"]
    cat = entry["cat"]
    
    reviews = [entry[k] for k in entry if k.startswith("rev")]
    summaries = [entry[k] for k in entry if k.startswith("summ")]

    for review in reviews:
        records.append({"prod_id": prod_id, "cat": cat, "type": "review", "text": review})
    for summary in summaries:
        records.append({"prod_id": prod_id, "cat": cat, "type": "summary", "text": summary})

df = pd.DataFrame(records)

# Word count distribution
df["word_count"] = df["text"].apply(lambda x: len(x.split()))
plt.figure()
df.boxplot(column="word_count", by="type")
plt.title("Word Count by Type")
plt.suptitle("")
plt.ylabel("Word Count")
plt.xlabel("Entry Type")
plt.savefig("word_count_by_type.png")  # Save to file
plt.close()

# WordCloud for all reviews
all_reviews = " ".join(df[df["type"] == "review"]["text"])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of All Reviews")
plt.savefig("review_wordcloud.png")  # Save to file
plt.close()
