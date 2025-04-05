import json

file_path = 'yelp_academic_dataset_review.json'

# Đếm số lượng sample và in ra sample đầu tiên
count = 0
first_sample = None

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if count == 0:
            first_sample = data
        count += 1

print(f'Tổng số sample: {count}')
print('Dữ liệu đầu tiên:')
print(json.dumps(first_sample, indent=2, ensure_ascii=False))

output_file = 'yelp_reviews_1M.json'
max_samples = 1000000

data_list = []

with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= max_samples:
            break
        data = json.loads(line)
        data_list.append(data)

with open(output_file, 'w', encoding='utf-8') as f_out:
    json.dump(data_list, f_out, ensure_ascii=False, indent=2)

print(f'Đã lưu {max_samples} dòng đầu tiên vào {output_file}')
