import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from basic_model import DualEncoderDecoderModel
import json

# Load dữ liệu
with open("../../data_building/models/sampling/sampled_OAs.json", "r") as f:
    sampled_oas = json.load(f)
with open("../../data_building/models/sampling/sampled_ISs.json", "r") as f:
    sampled_iss = json.load(f)
with open("../../data_building/models/results/pseudo_summary.json", "r") as f:
    pseudo_summary = json.load(f)

with open("../../data_building/models/results/pseudo_summary.json", "r") as f:
    dev_data = json.load(f)

# print('sampled_iss', len(sampled_iss))
# print('popular_pairs', len(sampled_oas['popular_pairs']))
# print('unpopular_pairs',len(sampled_oas['unpopular_pairs']))

def pairs_to_sentences(pairs):
    return [f"{pair[0]} [SEP] {pair[1]}" for pair in pairs]

# Hàm token hóa
def tokenize_texts(texts, tokenizer, max_seq_len, device):
    tokenized = tokenizer(
        texts, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt"
    )
    return tokenized["input_ids"].to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
max_seq_len = 256
vocab_size = 30522
d_model = 512
nhead = 8
num_layers = 6
dim_feedforward = 2048
learning_rate = 1e-4
num_epochs = 10

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_special_tokens({'additional_special_tokens': ['[SEP]']})

sampled_oa_pairs = sampled_oas["popular_pairs"] + sampled_oas["unpopular_pairs"]
oa_sentences = pairs_to_sentences(sampled_oa_pairs)
oa_input = tokenize_texts(oa_sentences, tokenizer, max_seq_len, device)

is_input = tokenize_texts(sampled_iss, tokenizer, max_seq_len, device)

min_len = min(len(oa_input), len(is_input))
oa_input = oa_input[:min_len]
is_input = is_input[:min_len]

tgt_input_ids = tokenize_texts([pseudo_summary["text"]], tokenizer, max_seq_len, device)
tgt_input_ids = tgt_input_ids.expand(oa_input.size(0), -1)

model = DualEncoderDecoderModel(
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dim_feedforward=dim_feedforward,
    vocab_size=vocab_size,
    max_seq_len=max_seq_len
).to(device)

# model.resize_token_embeddings(len(tokenizer))

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.train()
batch_size = 64
num_batches = len(oa_input) // batch_size + (len(oa_input) % batch_size != 0)

for epoch in range(num_epochs):
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(oa_input))
        
        if batch_end - batch_start < batch_size:
            # print(f"Skipping small batch of size {batch_end - batch_start}")
            continue

        batch_oa_input = oa_input[batch_start:batch_end]
        batch_is_input = is_input[batch_start:batch_end]
        batch_tgt_input_ids = tgt_input_ids[batch_start:batch_end]

        if torch.isnan(batch_oa_input).any() or torch.isnan(batch_is_input).any():
            # print("Detected NaN in inputs! Skipping batch.")
            continue

        optimizer.zero_grad()
        decoder_input = batch_tgt_input_ids[:, :-1]
        decoder_target = batch_tgt_input_ids[:, 1:]

        # Forward pass
        output = model(oa_input=batch_oa_input, is_input=batch_is_input, tgt_input=decoder_input)

        if torch.isnan(output).any():
            # print("NaN detected in model output")
            # print(f"Output min: {output.min().item()}, Output max: {output.max().item()}")
            continue

        output = torch.clamp(output, min=-1e4, max=1e4)
        output = output.view(-1, vocab_size)
        decoder_target = decoder_target.reshape(-1)
        loss = criterion(output, decoder_target)

        if torch.isnan(loss).any():
            # print("Detected NaN in Loss! Skipping batch.")
            continue

        loss.backward()
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any():
        #             print(f"NaN detected in gradients of {name}")
        #         print(f"Gradient stats for {name}: min={param.grad.min()}, max={param.grad.max()}")
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "dual_encoder_decoder_model_trained.pth")
