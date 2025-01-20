import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
from transformers import AutoTokenizer, AutoModel
import torch.optim as optim

class DualEncoderDecoderModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, vocab_size, max_seq_len):
        super(DualEncoderDecoderModel, self).__init__()
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer encoders for OA and IS
        self.oa_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=num_layers
        )
        self.is_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=num_layers
        )
        
        # Context vectors for OA and IS
        self.v_oa = nn.Parameter(torch.randn(d_model))
        self.v_is = nn.Parameter(torch.randn(d_model))
        
        # Transformer decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=num_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, oa_input, is_input, tgt_input):
        # Ensure inputs are LongTensors
        oa_input = oa_input.long()
        is_input = is_input.long()
        tgt_input = tgt_input.long()

        # Generate positional encodings
        oa_pos = self.positional_encoding(torch.arange(oa_input.size(1), device=oa_input.device)).unsqueeze(0)
        is_pos = self.positional_encoding(torch.arange(is_input.size(1), device=is_input.device)).unsqueeze(0)
        tgt_pos = self.positional_encoding(torch.arange(tgt_input.size(1), device=tgt_input.device)).unsqueeze(0)

        # Apply embeddings and positional encodings
        oa_input = self.embedding(oa_input) + oa_pos
        is_input = self.embedding(is_input) + is_pos
        tgt_input = self.embedding(tgt_input) + tgt_pos

        # Pass inputs through encoders
        ho = self.oa_encoder(oa_input)  # Shape: [batch_size, seq_len, d_model]
        hi = self.is_encoder(is_input)  # Shape: [batch_size, seq_len, d_model]

        # Aggregation
        ao = torch.mean(ho, dim=1)  # [batch_size, d_model]
        ai = torch.mean(hi, dim=1)  # [batch_size, d_model]

        # Compute attention weights (element-wise dot product)
        lambda_oa = torch.exp(torch.sum(ao * self.v_oa, dim=1)) / (
            torch.exp(torch.sum(ao * self.v_oa, dim=1)) + torch.exp(torch.sum(ai * self.v_is, dim=1))
        )
        lambda_is = 1 - lambda_oa

        # Expand dimensions for broadcasting
        lambda_oa = lambda_oa.unsqueeze(1).unsqueeze(-1)  # Shape: [batch_size, 1, 1]
        lambda_is = lambda_is.unsqueeze(1).unsqueeze(-1)  # Shape: [batch_size, 1, 1]

        # Weighted context aggregation
        co_t = lambda_oa * ho  # Shape: [batch_size, seq_len, d_model]
        ci_t = lambda_is * hi  # Shape: [batch_size, seq_len, d_model]
        ct = torch.cat((co_t, ci_t), dim=1)  # Shape: [batch_size, 2 * seq_len, d_model]

        # Pass through decoder
        decoder_output = self.decoder(tgt_input, memory=ct)

        # Output layer
        output = self.output_layer(decoder_output)
        return output


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_seq_len = 128
d_model = 512
vocab_size = 30522

# Load data
with open("../../data_building/models/sampling/sampled_ISs.json", "r") as f:
    sampled_is = json.load(f)

with open("../../data_building/models/sampling/sampled_OAs.json", "r") as f:
    sampled_oas = json.load(f)

with open("../../data_building/models/results/pseudo_summary.json", "r") as f:
    pseudo_summary = json.load(f)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Preprocess inputs
def tokenize_texts(texts, tokenizer, max_seq_len, device):
    tokenized = tokenizer(
        texts, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt"
    )
    return tokenized["input_ids"].to(device)

# Ensure consistent batch size
min_size = min(len(sampled_oas["popular_pairs"]), len(sampled_is))
sampled_oa_pairs = sampled_oas["popular_pairs"][:min_size]
sampled_is = sampled_is[:min_size]

# Tokenize inputs
oa_input = tokenize_texts(
    [" ".join(pair) for pair in sampled_oa_pairs],
    tokenizer,
    max_seq_len,
    device,
)
is_input = tokenize_texts(sampled_is, tokenizer, max_seq_len, device)
tgt_input_ids = tokenize_texts([pseudo_summary["text"]], tokenizer, max_seq_len, device)
tgt_input_ids = tgt_input_ids.expand(oa_input.size(0), -1)  # Match batch size of target with inputs

# Initialize model
model = DualEncoderDecoderModel(
    d_model=d_model,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048,
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
).to(device)

# Training parameters
learning_rate = 1e-4
num_epochs = 10
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Prepare decoder inputs and targets
    decoder_input = tgt_input_ids[:, :-1]  # Exclude last token
    decoder_target = tgt_input_ids[:, 1:]  # Exclude first token

    # Forward pass
    output = model(oa_input, is_input, decoder_input)

    # Reshape for loss computation
    output = output.view(-1, vocab_size)
    decoder_target = decoder_target.reshape(-1)

    # Debugging: Check values
    print(f"Decoder target min: {decoder_target.min()}, max: {decoder_target.max()}")
    print(f"Output min: {output.min()}, max: {output.max()}")

    # Clamp logits
    output = torch.clamp(output, min=-1e10, max=1e10)

    # Compute loss
    loss = criterion(output, decoder_target)

    # Check for NaN loss
    if torch.isnan(loss):
        print("NaN loss encountered. Stopping training.")
        break

    # Backward pass
    loss.backward()

    # Debugging: Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN detected in gradients of {name}")

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimization step
    optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "dual_encoder_decoder_model.pth")
print("Model saved to dual_encoder_decoder_model.pth")
