import pandas as pd
import json

# Đường dẫn đến file CSV
file_path = '../../../data/yelp/test/summaries_0-200_cleaned.csv'
data = pd.read_csv(file_path)

# Chia dữ liệu thành dev_data và test_data
dev_data = data.iloc[:100]
test_data = data.iloc[100:200]

# Hàm xử lý để chuyển dataframe thành định dạng JSON
def process_to_json(dataframe):
    processed_data = []
    for _, row in dataframe.iterrows():
        reviews = {f"review_{i+1}": row[f"Input.original_review_{i}"] for i in range(8)}
        summary = row["Answer.summary"]
        processed_data.append({"reviews": reviews, "summary": summary})
    return processed_data

# Xử lý dữ liệu dev và test
dev_json = process_to_json(dev_data)
test_json = process_to_json(test_data)

# Lưu kết quả vào file JSON
dev_json_path = 'results/dev_data.json'
test_json_path = 'results/test_data.json'

with open(dev_json_path, 'w') as f:
    json.dump(dev_json, f, indent=4)
with open(test_json_path, 'w') as f:
    json.dump(test_json, f, indent=4)