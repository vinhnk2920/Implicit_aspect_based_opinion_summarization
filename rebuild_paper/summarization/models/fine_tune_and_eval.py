import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm
from rouge_score import rouge_scorer
from research.Implicit_aspect_based_opinion_summarization.rebuild_paper.summarization.models.basic_model_0 import DualEncoderDecoderModel


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len, device):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_ids = self.tokenizer(
            sample["input"], truncation=True, padding="max_length", max_length=self.max_seq_len, return_tensors="pt"
        )["input_ids"].squeeze(0)
        output_ids = self.tokenizer(
            sample["output"], truncation=True, padding="max_length", max_length=self.max_seq_len, return_tensors="pt"
        )["input_ids"].squeeze(0)
        return input_ids.to(self.device), output_ids.to(self.device)


# Function to load data
def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# Function to evaluate and compute ROUGE
def evaluate_and_save_results(model, data_loader, tokenizer, device, output_file):
    model.eval()
    predictions, references = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating"):
            decoder_input = targets[:, :-1]
            outputs = model(inputs, inputs, decoder_input)
            
            # Decode predictions and references
            predicted_texts = tokenizer.batch_decode(
                torch.argmax(outputs, dim=-1), skip_special_tokens=True
            )
            reference_texts = tokenizer.batch_decode(targets, skip_special_tokens=True)
            
            # Debugging
            print(f"Predicted token IDs: {torch.argmax(outputs, dim=-1)[0]}")  # Token IDs
            print(f"Decoded prediction: {predicted_texts[0]}")  # First decoded output
            print(f"Decoded reference: {reference_texts[0]}")  # First reference
            
            predictions.extend(predicted_texts)
            references.extend(reference_texts)

    # Save predictions and references
    with open(output_file, "w") as f:
        json.dump({"predictions": predictions, "references": references}, f, indent=4)
    print(f"Results saved to {output_file}")

    # Compute ROUGE scores
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(pred, ref)
        for key in rouge_scores.keys():
            rouge_scores[key].append(scores[key].fmeasure)

    avg_rouge = {key: sum(vals) / len(vals) for key, vals in rouge_scores.items()}
    print(f"Average ROUGE Scores: {avg_rouge}")
    return avg_rouge



# Main fine-tuning and evaluation function
def fine_tune_and_evaluate(
    dev_file, test_file, model, tokenizer, max_seq_len, device, batch_size=8, epochs=5
):
    # Load datasets
    train_data = load_data(dev_file)
    test_data = load_data(test_file)

    # Create DataLoaders
    train_loader = DataLoader(
        TextDataset(train_data, tokenizer, max_seq_len, device),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TextDataset(test_data, tokenizer, max_seq_len, device),
        batch_size=batch_size,
        shuffle=False,
    )

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Fine-tuning loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            decoder_input = targets[:, :-1]
            decoder_target = targets[:, 1:]
            outputs = model(inputs, inputs, decoder_input)
            # loss = criterion(outputs.view(-1, tokenizer.vocab_size), decoder_target.view(-1))
            loss = criterion(outputs.reshape(-1, tokenizer.vocab_size), decoder_target.reshape(-1))
            print(f"Outputs shape: {outputs.shape}")  # Should be [batch_size, seq_len, vocab_size]
            print(f"Decoder target shape: {decoder_target.shape}")  # Should be [batch_size, seq_len]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    # Save the model
    torch.save(model.state_dict(), "fine_tuned_model.pth")
    print("Model saved to fine_tuned_model.pth")

    # Evaluate on test data
    output_file = "fine_tuned_test_results.json"
    avg_rouge = evaluate_and_save_results(model, test_loader, tokenizer, device, output_file)
    print(f"Test Results saved to {output_file}")
    print(f"Average ROUGE Scores: {avg_rouge}")


# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_seq_len = 128
batch_size = 8
epochs = 5

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Initialize model
model = DualEncoderDecoderModel(
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048,
    vocab_size=tokenizer.vocab_size,
    max_seq_len=max_seq_len,
).to(device)

# File paths
dev_file = "../data_preparation/results/dev_data.json"
test_file = "../data_preparation/results/test_data.json"

# Fine-tune and evaluate
fine_tune_and_evaluate(dev_file, test_file, model, tokenizer, max_seq_len, device, batch_size, epochs)