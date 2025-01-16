import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BartForConditionalGeneration
import torch.nn as nn
import torch.optim as optim

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

oas_file_path = "../../data_building/outputs/sampled_oas.json"
iss_file_path = "../../data_building/outputs/sampled_iss.json"
review_summary_path = "../data_preparation/results/test_data.json"

oas_data = load_json(oas_file_path)
iss_data = load_json(iss_file_path)
review_summary_data = load_json(review_summary_path)  # Dữ liệu test review-summary

class EncoderDataset(Dataset):
    def __init__(self, oas, iss, tokenizer, max_length=128):
        self.oas = oas
        self.iss = iss
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.oas)

    def __getitem__(self, idx):
        oas_encoded = self.tokenizer(self.oas[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        iss_encoded = self.tokenizer(self.iss[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {
            "oas_input_ids": oas_encoded["input_ids"].squeeze(0),
            "oas_attention_mask": oas_encoded["attention_mask"].squeeze(0),
            "iss_input_ids": iss_encoded["input_ids"].squeeze(0),
            "iss_attention_mask": iss_encoded["attention_mask"].squeeze(0)
        }

class ReviewSummaryDataset(Dataset):
    def __init__(self, reviews, summaries, tokenizer, max_length=128):
        self.reviews = reviews
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review_encoded = self.tokenizer(self.reviews[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        summary_encoded = self.tokenizer(self.summaries[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {
            "review_input_ids": review_encoded["input_ids"].squeeze(0),
            "review_attention_mask": review_encoded["attention_mask"].squeeze(0),
            "summary_input_ids": summary_encoded["input_ids"].squeeze(0),
            "summary_attention_mask": summary_encoded["attention_mask"].squeeze(0)
        }

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

oas_sentences = [f"{item['aspect']}: {item['opinion']}" for item in oas_data["popular"]]
iss_sentences = [sentence for sentence in iss_data]

# Align OA and IS datasets for parallelism
max_len = min(len(oas_sentences), len(iss_sentences))
oas_sentences = oas_sentences[:max_len]
iss_sentences = iss_sentences[:max_len]

oas_iss_dataset = EncoderDataset(oas_sentences, iss_sentences, tokenizer)
review_sentences = [item["input"] for item in review_summary_data]
summary_sentences = [item["output"] for item in review_summary_data]
review_summary_dataset = ReviewSummaryDataset(review_sentences, summary_sentences, tokenizer)

oas_iss_dataloader = DataLoader(oas_iss_dataset, batch_size=5, shuffle=True)
review_summary_dataloader = DataLoader(review_summary_dataset, batch_size=5, shuffle=False)

# 3. Model Definition
class DualEncoderModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers):
        super(DualEncoderModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # OA and IS encoders
        self.oa_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim),
            num_layers=num_layers
        )
        self.is_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim),
            num_layers=num_layers
        )

        # Attention weights for combining OA and IS
        self.oa_attention = nn.Linear(embed_dim, 1)
        self.is_attention = nn.Linear(embed_dim, 1)

        # Projection layer to match BART hidden size
        self.projection = nn.Linear(embed_dim, 1024)

        # Full BART model (with decoder and lm_head)
        self.bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

    def forward(self, oas_inputs, iss_inputs, decoder_input_ids):
        # Embedding inputs
        oas_embedded = self.embedding(oas_inputs)  # (batch_size, seq_len, embed_dim)
        iss_embedded = self.embedding(iss_inputs)  # (batch_size, seq_len, embed_dim)

        # Encode OA and IS
        oas_encoded = self.oa_encoder(oas_embedded)  # (batch_size, seq_len, embed_dim)
        iss_encoded = self.is_encoder(iss_embedded)  # (batch_size, seq_len, embed_dim)

        # Compute attention weights
        oas_weights = torch.softmax(self.oa_attention(oas_encoded), dim=1)  # (batch_size, seq_len, 1)
        iss_weights = torch.softmax(self.is_attention(iss_encoded), dim=1)  # (batch_size, seq_len, 1)

        # Weighted sum of encoded outputs
        oas_context = torch.sum(oas_weights * oas_encoded, dim=1)  # (batch_size, embed_dim)
        iss_context = torch.sum(iss_weights * iss_encoded, dim=1)  # (batch_size, embed_dim)

        # Combine contexts
        combined_context = oas_context + iss_context

        # Project to match BART's hidden size
        projected_context = self.projection(combined_context)  # (batch_size, 1024)

        # Decode using BART's full model
        outputs = self.bart_model(
            input_ids=decoder_input_ids,
            encoder_outputs=(projected_context.unsqueeze(1),),  # Pass combined context as encoder output
            return_dict=True,
        )

        return outputs.logits


# Initialize model
EMBED_DIM = 768
NUM_HEADS = 8
FF_DIM = 2048
NUM_LAYERS = 6
VOCAB_SIZE = tokenizer.vocab_size

model = DualEncoderModel(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=5e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in review_summary_dataloader:
        review_input_ids = batch["review_input_ids"].to(device)
        summary_input_ids = batch["summary_input_ids"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(review_input_ids, review_input_ids, summary_input_ids)
        shift_logits = outputs[:, :-1, :].contiguous()
        shift_labels = summary_input_ids[:, 1:].contiguous()

        # Compute loss
        loss = criterion(shift_logits.view(-1, VOCAB_SIZE), shift_labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(review_summary_dataloader)}")

# Save the model and results
torch.save(model.state_dict(), "dual_encoder_model_weights.pth")
print("Model weights saved to dual_encoder_model_weights.pth")

# Evaluation and saving results
model.eval()
results = []

with torch.no_grad():
    for batch in review_summary_dataloader:
        review_input_ids = batch["review_input_ids"].to(device)
        summary_input_ids = batch["summary_input_ids"].to(device)

        # Generate predictions
        predicted_ids = model.bart_model.generate(
            input_ids=review_input_ids,
            attention_mask=batch["review_attention_mask"].to(device),
            max_length=128,
            num_beams=4
        )

        # Decode predictions and references
        review_texts = tokenizer.batch_decode(review_input_ids, skip_special_tokens=True)
        predicted_summaries = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        target_summaries = tokenizer.batch_decode(summary_input_ids, skip_special_tokens=True)

        # Save results
        for review, predicted, target in zip(review_texts, predicted_summaries, target_summaries):
            results.append({
                "review": review,
                "predicted_summary": predicted,
                "target_summary": target
            })

# Save results to a JSON file
output_file = "review_summary_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_file}")
