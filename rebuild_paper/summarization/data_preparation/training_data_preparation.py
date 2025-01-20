import json

def prepare_train_data(opinion_file, implicit_file, output_train_file):
    with open(opinion_file, 'r') as f:
        opinion_data = json.load(f)
    with open(implicit_file, 'r') as f:
        implicit_data = json.load(f)

    popular_aspects = [
        {"aspect": pair["aspect"], "opinion": pair["opinion"]}
        for pair in opinion_data.get("popular", [])
    ]
    unpopular_aspects = opinion_data.get("unpopular", [])

    # Tạo tập dữ liệu huấn luyện
    train_data = []

    # Xử lý OA-pairs
    train_data.extend([
        {
            "type": "oa_pairs",
            "input": {
                "popular": popular_aspects,
                "unpopular": unpopular_aspects
            }
        }
    ])

    # Xử lý Implicit Sentences
    for sentence in implicit_data:
        train_data.append({
            "type": "implicit_sentence",
            "input": sentence
        })

    # Lưu dữ liệu huấn luyện vào file JSON
    with open(output_train_file, 'w') as f:
        json.dump(train_data, f, indent=4)

    print(f"Train data saved to {output_train_file}")

prepare_train_data(
    opinion_file='../../data_building/outputs/sampled_oas.json',
    implicit_file='../../data_building/outputs/sampled_iss.json',
    output_train_file='train_data.json'
)