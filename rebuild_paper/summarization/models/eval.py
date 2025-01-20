import torch
import json
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from research.Implicit_aspect_based_opinion_summarization.rebuild_paper.summarization.models.basic_model_0 import DualEncoderDecoderModel
from tqdm import tqdm

# Hàm khởi tạo mô hình
def load_model(model_path, d_model, vocab_size, max_seq_len, device):
    model = DualEncoderDecoderModel(
        d_model=d_model,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

# Hàm token hóa dữ liệu
def tokenize_text(text, tokenizer, max_seq_len, device):
    tokenized = tokenizer(
        text, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt"
    )
    return tokenized["input_ids"].to(device)

# Hàm tính ROUGE trung bình
def calculate_average_rouge(results):
    rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for res in results:
        scores = rouge_scorer_obj.score(res["expected"], res["predicted"])
        for key in scores:
            rouge_scores[key].append(scores[key].fmeasure)

    average_rouge = {key: sum(values) / len(values) for key, values in rouge_scores.items()}
    return average_rouge

# Hàm thực hiện dự đoán
def predict_and_evaluate(model, test_data, tokenizer, max_seq_len, device):
    results = []
    for sample in tqdm(test_data, desc="Processing test data"):
        try:
            # Tokenize inputs
            oa_input = tokenize_text(sample["input"], tokenizer, max_seq_len, device)
            is_input = tokenize_text(sample["input"], tokenizer, max_seq_len, device)
            tgt_input = tokenize_text(sample["output"], tokenizer, max_seq_len, device)

            # Chuẩn bị input cho decoder
            decoder_input = tgt_input[:, :-1]  # Exclude last token

            # Predict output
            with torch.no_grad():
                output = model(oa_input, is_input, decoder_input)
                predicted_ids = torch.argmax(output, dim=-1)

            # Decode dự đoán
            predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

            # Lưu kết quả
            results.append({
                "input": sample["input"],
                "predicted": predicted_text,
                "expected": sample["output"]
            })
        except Exception as e:
            print(f"Lỗi xử lý mẫu: {sample}. Chi tiết lỗi: {e}")
    return results

# Main script
if __name__ == "__main__":
    # Cấu hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "dual_encoder_decoder_model.pth"
    test_data_path = "../data_preparation/results/test_data.json"
    output_results_path = "predicted_results.json"
    d_model = 512
    vocab_size = 30522
    max_seq_len = 128

    # Load mô hình và tokenizer
    model = load_model(model_path, d_model, vocab_size, max_seq_len, device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load dữ liệu test
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    # Thực hiện dự đoán và tính ROUGE
    results = predict_and_evaluate(model, test_data, tokenizer, max_seq_len, device)

    # Lưu kết quả
    with open(output_results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Kết quả đã được lưu vào {output_results_path}")

    # Tính ROUGE trung bình
    average_rouge = calculate_average_rouge(results)
    print("ROUGE trung bình:", average_rouge)
