import json
from training import tokenize_texts
from transformers import AutoTokenizer
import torch
from basic_model import DualEncoderDecoderModel
# from rouge import Rouge  # Import Rouge library

# Load validation data
with open("../data_preparation/results/test_data.json", "r") as f:
    validation_data = json.load(f)

# Chuẩn bị dữ liệu
validation_reviews = [
    " ".join(data["reviews"].values()) for data in validation_data
]
validation_summaries = [
    data["summary"] for data in validation_data
]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
max_seq_len = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 30522

# Tokenize validation data
print('Tokenize validation_reviews')
validation_inputs = tokenize_texts(validation_reviews, tokenizer, max_seq_len, device)
print('review 0: ', len(validation_reviews[0]))
print('review 1: ', len(validation_reviews[1]))
print('review 2: ', len(validation_reviews[2]))
print('review 3: ', len(validation_reviews[3]))
print('review 4: ', len(validation_reviews[4]))
print('review 5: ', len(validation_reviews[5]))
print('===================================================================')
print('Tokenize validation_summaries')
print('summary 0: ', len(validation_summaries[0]))
print('summary 1: ', len(validation_summaries[1]))
print('summary 2: ', len(validation_summaries[2]))
print('summary 3: ', len(validation_summaries[3]))
print('summary 4: ', len(validation_summaries[4]))
print('summary 5: ', len(validation_summaries[5]))
validation_targets = tokenize_texts(validation_summaries, tokenizer, max_seq_len, device)


d_model = 512
nhead = 8
num_layers = 6
dim_feedforward = 2048
vocab_size = 30522
max_seq_len = 256

# Khởi tạo mô hình
model = DualEncoderDecoderModel(
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dim_feedforward=dim_feedforward,
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
)

model.load_state_dict(torch.load("dual_encoder_decoder_model_trained.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

def evaluate_model(model, inputs, targets, criterion, tokenizer, batch_size=64):
    model.eval()
    total_loss = 0
    num_batches = len(inputs) // batch_size + (len(inputs) % batch_size != 0)

    predictions = []
    references = []

    with torch.no_grad():  # Tắt tính toán gradient
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(inputs))

            batch_inputs = inputs[batch_start:batch_end]
            batch_targets = targets[batch_start:batch_end]

            decoder_input = batch_targets[:, :-1]
            decoder_target = batch_targets[:, 1:]

            output = model(oa_input=batch_inputs, is_input=batch_inputs, tgt_input=decoder_input)

            output = output.view(-1, output.size(-1))
            decoder_target = decoder_target.reshape(-1)
            loss = criterion(output, decoder_target)
            total_loss += loss.item()

            predicted_tokens = torch.argmax(output, dim=-1)
            for i in range(batch_inputs.size(0)):
                pred = tokenizer.decode(predicted_tokens[i], skip_special_tokens=True).strip()
                ref = tokenizer.decode(batch_targets[i], skip_special_tokens=True).strip()

                predictions.append(pred)
                references.append(ref)

    avg_loss = total_loss / num_batches
    return avg_loss, predictions, references

if __name__ == "__main__":
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    val_loss, predictions, references = evaluate_model(
        model, validation_inputs, validation_targets, criterion, tokenizer
    )

    print(f"Validation Loss: {val_loss:.4f}")

    output_results = {
        "Validation Loss": val_loss,
        "Predictions": predictions,
        "References": references
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(output_results, f, indent=4)

    # # Lọc các cặp hợp lệ để tính ROUGE
    # valid_predictions = []
    # valid_references = []
    # for pred, ref in zip(predictions, references):
    #     if pred and ref:  # Chỉ giữ các chuỗi không rỗng
    #         valid_predictions.append(pred)
    #         valid_references.append(ref)

    # if valid_predictions and valid_references:
    #     rouge = Rouge()
    #     rouge_scores = rouge.get_scores(valid_predictions, valid_references, avg=True)
    #     print("Average ROUGE Scores:")
    #     print(rouge_scores)

        # Lưu ROUGE Scores vào file
        # with open("rouge_scores.json", "w") as f:
        #     json.dump(rouge_scores, f, indent=4)
    # else:
    #     print("No valid predictions or references for ROUGE calculation.")
