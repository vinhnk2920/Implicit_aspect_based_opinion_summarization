import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
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
    def __init__(self, sentences, tokenizer, max_length=128):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoded = self.tokenizer(self.sentences[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
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


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

oas_sentences = [f"{item['aspect']}: {item['opinion']}" for item in oas_data["popular"]]
iss_sentences = [sentence for sentence in iss_data]

oas_dataset = EncoderDataset(oas_sentences, tokenizer)
iss_dataset = EncoderDataset(iss_sentences, tokenizer)

review_sentences = [item["input"] for item in review_summary_data]
summary_sentences = [item["output"] for item in review_summary_data]
review_summary_dataset = ReviewSummaryDataset(review_sentences, summary_sentences, tokenizer)

oas_dataloader = DataLoader(oas_dataset, batch_size=5, shuffle=True)
iss_dataloader = DataLoader(iss_dataset, batch_size=5, shuffle=True)
review_summary_dataloader = DataLoader(review_summary_dataset, batch_size=5, shuffle=False)

# 3. Model Definition (same as before)
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, vocab_size):
        super(TransformerDecoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory):
        decoded = self.transformer_decoder(tgt, memory)
        return self.output_layer(decoded)

class BasicModel(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, vocab_size):
        super(BasicModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.oa_encoder = TransformerEncoder(embed_dim, num_heads, ff_dim, num_layers)
        self.is_encoder = TransformerEncoder(embed_dim, num_heads, ff_dim, num_layers)
        self.decoder = TransformerDecoder(embed_dim, num_heads, ff_dim, num_layers, vocab_size)

    def forward(self, oa_inputs, is_inputs, tgt_inputs):
        oa_embedded = self.embedding(oa_inputs) 
        is_embedded = self.embedding(is_inputs) 
        tgt_embedded = self.embedding(tgt_inputs)

        # Encode OA and IS
        oa_encoded = self.oa_encoder(oa_embedded)
        is_encoded = self.is_encoder(is_embedded)

        # Combine context
        combined_context = oa_encoded + is_encoded

        # Decode
        output = self.decoder(tgt_embedded, combined_context)
        return output


# 4. Initialize Model
EMBED_DIM = 512
NUM_HEADS = 8
FF_DIM = 2048
NUM_LAYERS = 6
VOCAB_SIZE = tokenizer.vocab_size

model = BasicModel(EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, VOCAB_SIZE)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

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

results = []
model.eval()
with torch.no_grad():
    for batch in review_summary_dataloader:
        review_input_ids = batch["review_input_ids"].to(device)
        summary_input_ids = batch["summary_input_ids"].to(device)
        outputs = model(review_input_ids, review_input_ids, summary_input_ids)

        for i in range(review_input_ids.size(0)):
            predicted_summary = tokenizer.decode(outputs.argmax(dim=-1)[i], skip_special_tokens=True)
            target_summary = tokenizer.decode(summary_input_ids[i], skip_special_tokens=True)

            results.append({
                "review": tokenizer.decode(review_input_ids[i], skip_special_tokens=True),
                "predicted_summary": predicted_summary,
                "target_summary": target_summary
            })

with open("test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

torch.save(model.state_dict(), "basic_model_weights.pth")
print("Test results saved to test_results.json")
print("Model weights saved to basic_model_weights.pth")
