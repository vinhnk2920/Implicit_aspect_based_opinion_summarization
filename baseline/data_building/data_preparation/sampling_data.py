import pandas as pd

df = pd.read_json("../../../data/yelp/train/yelp_academic_dataset_review.json", lines=True)

# base on review length

df['review_length'] = df['text'].apply(lambda x: len(x.split()))
filtered_df = df[(df['stars'] > 4.0) & (df['review_length'] > 20) & (df['review_length'] < 200)]
print(filtered_df.head(2))
print(len(filtered_df))

sample_df = df.sample(n=300000, random_state=42)
print(len(sample_df))

sample_df.to_json("../../../data/yelp/train/yelp_train_300k.json", orient="records", lines=True)
