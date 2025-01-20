import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, AdamW, get_scheduler
from tqdm import tqdm


class OpinionDataset(Dataset):
    def __init__(self, data, tokenizer, oa_data, is_data, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.oa_data = oa_data
        self.is_data = is_data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize opinion-aspect (OA) pairs
        oa_input = self.tokenizer(
            " ".join([" ".join(pair) for pair in self.oa_data["popular_pairs"] + self.oa_data["unpopular_pairs"]]),
            truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        
        # Tokenize implicit sentences (IS)
        is_input = self.tokenizer(
            " ".join(self.is_data),
            truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        
        # Tokenize the ground truth summary
        summary = self.tokenizer(
            item["output"],
            truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        
        return {
            "oa_input": oa_input["input_ids"].squeeze(0),
            "oa_mask": oa_input["attention_mask"].squeeze(0),
            "is_input": is_input["input_ids"].squeeze(0),
            "is_mask": is_input["attention_mask"].squeeze(0),
            "labels": summary["input_ids"].squeeze(0),
        }


class Condense(nn.Module):
    def __init__(self, aspect_dim, sentiment_dim, input_dim, hidden_dim, vocab_size):
        super(Condense, self).__init__()
        self.oa_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8), num_layers=3
        )
        self.is_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8), num_layers=3
        )
        config = BartTokenizer.from_pretrained('facebook/bart-base').config
        config.vocab_size = vocab_size
        config.d_model = input_dim
        self.decoder = BartForConditionalGeneration(config)

    def forward(self, oa_input, is_input, oa_mask, is_mask):
        oa_encoded = self.oa_encoder(oa_input, src_key_padding_mask=oa_mask)
        is_encoded = self.is_encoder(is_input, src_key_padding_mask=is_mask)
        combined_representation = torch.cat([oa_encoded, is_encoded], dim=1)
        outputs = self.decoder(input_ids=None, encoder_outputs=(combined_representation,))
        return outputs.logits


def train_model(model, train_loader, dev_loader, optimizer, scheduler, tokenizer, num_epochs, device):
    model.train()
    best_loss = float("inf")

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            oa_input = batch["oa_input"].to(device)
            is_input = batch["is_input"].to(device)
            oa_mask = batch["oa_mask"].to(device)
            is_mask = batch["is_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(oa_input, is_input, oa_mask, is_mask)
            loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        dev_loss = evaluate_model(model, dev_loader, device)

        print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}, Dev Loss: {dev_loss:.4f}")

        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(model.state_dict(), "model_2.pt")
            print("Model saved!")


def evaluate_model(model, dev_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dev_loader:
            oa_input = batch["oa_input"].to(device)
            is_input = batch["is_input"].to(device)
            oa_mask = batch["oa_mask"].to(device)
            is_mask = batch["is_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(oa_input, is_input, oa_mask, is_mask)
            loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()
    return total_loss / len(dev_loader)


def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def main():
    # Load data
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    sampled_oa = load_data("sampled_OAs.json")
    sampled_is = load_data("sampled_ISs.json")
    train_data = load_data("train_data.json")
    dev_data = load_data("dev_data.json")

    # Prepare datasets and loaders
    train_dataset = OpinionDataset(train_data, tokenizer, sampled_oa, sampled_is)
    dev_dataset = OpinionDataset(dev_data, tokenizer, sampled_oa, sampled_is)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=8)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Condense(aspect_dim=128, sentiment_dim=128, input_dim=768, hidden_dim=512, vocab_size=len(tokenizer))
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=3e-5)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=200, num_training_steps=5000)

    # Train the model
    train_model(model, train_loader, dev_loader, optimizer, scheduler, tokenizer, num_epochs=5, device=device)


if __name__ == "__main__":
    main()
