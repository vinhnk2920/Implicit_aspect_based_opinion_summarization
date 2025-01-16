import pandas as pd

# Đọc tệp CSV
file_path = "summaries_0-200_cleaned.csv"
data = pd.read_csv(file_path)

# Kiểm tra số lượng dòng
print(f"Total records: {len(data)}")

dev_data = data.iloc[:100]
test_data = data.iloc[100:]

# Kiểm tra kết quả
print(f"Dev set size: {len(dev_data)}")
print(f"Test set size: {len(test_data)}")

dev_data.to_csv("summaries_dev.csv", index=False)
test_data.to_csv("summaries_test.csv", index=False)

print("Data has been split and saved successfully!")