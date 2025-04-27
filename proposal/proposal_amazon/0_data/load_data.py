import random

# Đường dẫn các file .jsonl
files = {
    'Clothing_Shoes_and_Jewelry': './results/raw/Clothing_Shoes_and_Jewelry.jsonl',
    'Electronics': './results/raw/Electronics.jsonl',
    'Health_and_Personal_Care': './results/raw/Health_and_Personal_Care.jsonl',
    'Home_and_Kitchen': './results/raw/Home_and_Kitchen.jsonl'
}

# Số lượng mẫu cần lấy từ mỗi category theo tỷ lệ test set
samples_needed = {
    'Clothing_Shoes_and_Jewelry': 133333,
    'Electronics': 133333,
    'Health_and_Personal_Care': 133333,
    'Home_and_Kitchen': 100000
}

# Mảng để lưu toàn bộ 500k dữ liệu mới
all_sampled_data = []

# Tiến hành sampling theo từng category
for category, file_path in files.items():
    print(f'Processing {category}...')
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Random sampling
    sampled_lines = random.sample(lines, samples_needed[category])

    # Thêm vào danh sách chung
    all_sampled_data.extend(sampled_lines)

# Lưu toàn bộ 500k vào 1 file mới
output_file = './results/training/amazon_training_500k.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(all_sampled_data)

print(f'Successfully saved 500,000 samples to {output_file}')
