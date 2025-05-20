import json
from datasets import Dataset
import os
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import sys


class DualEncoderBART(nn.Module):
    def __init__(self, bart_model_name="facebook/bart-large"):
        print("Initializing basic model...")
        super(DualEncoderBART, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained(bart_model_name)

        self.bart_oa = BartForConditionalGeneration.from_pretrained(
            bart_model_name,
            dropout=0.1,
            attention_dropout=0.1
        )
        self.bart_is = BartForConditionalGeneration.from_pretrained(
            bart_model_name,
            dropout=0.1,
            attention_dropout=0.1
        )

        hidden_size = self.bart_oa.config.d_model
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, dropout=0.1)
        self.v_o = nn.Parameter(0.01 * torch.randn(hidden_size, 1, requires_grad=True))
        self.v_i = nn.Parameter(0.01 * torch.randn(hidden_size, 1, requires_grad=True))
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, oas_input, iss_input, decoder_input):
        oa_encoder = self.bart_oa.get_encoder()
        oa_outputs = oa_encoder(input_ids=oas_input["input_ids"], attention_mask=oas_input["attention_mask"])
        ho = oa_outputs.last_hidden_state

        is_encoder = self.bart_is.get_encoder()
        is_outputs = is_encoder(input_ids=iss_input["input_ids"], attention_mask=iss_input["attention_mask"])
        hi = is_outputs.last_hidden_state

        min_seq_len = min(ho.size(1), hi.size(1))
        ho, hi = ho[:, :min_seq_len, :], hi[:, :min_seq_len, :]

        exp_ao = torch.softmax(torch.matmul(ho, self.v_o) / self.temperature, dim=-1)
        exp_ai = torch.softmax(torch.matmul(hi, self.v_i) / self.temperature, dim=-1)
        # print(f"exp_ao.shape: {exp_ao.shape}")
        # print(f"exp_ai.shape: {exp_ai.shape}")
        lambda_o = exp_ao / (exp_ao + exp_ai)
        lambda_i = 1 - lambda_o
        # print(f"lambda_o.shape: {lambda_o.shape}, lambda_i.shape: {lambda_i.shape}")  
        # print("Lambda_o:", lambda_o.mean().item(), "Lambda_i:", lambda_i.mean().item())
        # print(f"lambda_o.shape: {lambda_o.shape}")  # Kỳ vọng: [4, 20, 1]
        # print(f"ho.shape: {ho.shape}")
        ct = lambda_o * ho + lambda_i * hi
        # print(f"ct.shape: {ct.shape}")
        # print("CT mean:", ct.mean().item(), "CT std:", ct.std().item())
        # print("Temperature:", self.temperature.item())
        max_decoder_len = min(ct.shape[1], decoder_input.shape[1])
        decoder_input = decoder_input[:, :max_decoder_len]
        max_encoder_len = ct.shape[1]  # Số lượng token hợp lệ từ encoder
        attention_mask = oas_input["attention_mask"][:, :max_encoder_len]
        # print(f"decoder_input.shape: {decoder_input.shape}")  
        # print(f"oas_input['attention_mask'].shape: {oas_input['attention_mask'].shape}")
        # print(f"Updated attention_mask.shape: {attention_mask.shape}")

        decoder_outputs = self.bart_oa(
            input_ids=decoder_input,
            encoder_outputs=BaseModelOutput(last_hidden_state=ct),
            attention_mask=attention_mask
        )
        # print("Decoder output logits mean:", decoder_outputs.logits.mean().item())
        # print("Decoder output logits std:", decoder_outputs.logits.std().item())

        return decoder_outputs.logits

    def generate(self, oas_input, iss_input, max_length=256, min_length=100, num_beams=5):
        """
        Generate a summary using the dual encoder.
        """
        # Encode Opinion Aspects (OA)
        oa_encoder = self.bart_oa.get_encoder()
        oa_outputs = oa_encoder(input_ids=oas_input["input_ids"], attention_mask=oas_input["attention_mask"])
        ho = oa_outputs.last_hidden_state

        # Encode Implicit Sentences (IS)
        is_encoder = self.bart_is.get_encoder()
        is_outputs = is_encoder(input_ids=iss_input["input_ids"], attention_mask=iss_input["attention_mask"])
        hi = is_outputs.last_hidden_state

        # Combine Context Vectors
        min_seq_len = min(ho.size(1), hi.size(1))
        ho, hi = ho[:, :min_seq_len, :], hi[:, :min_seq_len, :]
        exp_ao = torch.softmax(torch.matmul(ho, self.v_o) / self.temperature, dim=-1)
        exp_ai = torch.softmax(torch.matmul(hi, self.v_i) / self.temperature, dim=-1)
        lambda_o = exp_ao / (exp_ao + exp_ai)
        lambda_i = 1 - lambda_o
        print("Lambda_o:", lambda_o.mean().item(), "Lambda_i:", lambda_i.mean().item())
        ct = lambda_o.unsqueeze(-1) * ho + lambda_i.unsqueeze(-1) * hi
        print("CT mean:", ct.mean().item(), "CT std:", ct.std().item())

        encoder_output = BaseModelOutput(last_hidden_state=ct)
        outputs = self.bart_oa.generate(
            encoder_outputs=encoder_output,
            attention_mask=oas_input["attention_mask"],  
            max_length=max_length,
            min_length=min_length,
            # num_beams=num_beams,
            do_sample=True,
            early_stopping=True
        )

        generated_summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_summary

    def save(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, "model_state_dict.pt"))
        self.tokenizer.save_pretrained(save_directory)

    def load(self, load_directory):
        self.load_state_dict(torch.load(os.path.join(load_directory, "model_state_dict.pt")))
        self.tokenizer = BartTokenizer.from_pretrained(load_directory)

def collate_fn(batch):
    oas_texts = []
    iss_texts = []
    targets = []

    for item in batch:
        oas_text = " ".join([
            f"[OA] {aspect}: {opinion} with sentiment {json.loads(sentiment)['label']}" 
            if isinstance(sentiment, str) else f"[OA] {aspect}: {opinion} with sentiment {sentiment['label']}"
            for aspect, opinion, sentiment in item["input"]["oas"]
        ])
        iss_text = " ".join([
            f"[IS] {is_entry['text']} with sentiment {json.loads(is_entry['sentiment'])['label']}" 
            if isinstance(is_entry['sentiment'], str) else f"[IS] {is_entry['text']} with sentiment {is_entry['sentiment']['label']}"
            for is_entry in item["input"]["iss"]
        ])
        oas_texts.append(oas_text)
        iss_texts.append(iss_text)
        targets.append(item["summary"])

    # Tokenize and pad inputs
    oas_inputs = tokenizer(oas_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    iss_inputs = tokenizer(iss_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    target_inputs = tokenizer(targets, return_tensors="pt", padding=True, truncation=True, max_length=512)

    return {
        "oas_input": oas_inputs,
        "iss_input": iss_inputs,
        "decoder_input": target_inputs["input_ids"],
        "labels": target_inputs["input_ids"]
    }

def train_model(model, dataloader, optimizer, num_epochs, device):
    model.to(device)
    model.train()
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            oas_input = {key: val.to(device) for key, val in batch["oas_input"].items()}
            iss_input = {key: val.to(device) for key, val in batch["iss_input"].items()}
            decoder_input = batch["decoder_input"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(oas_input, iss_input, decoder_input)
            seq_len = outputs.shape[1]  # Lấy seq_len của decoder output
            labels = labels[:, :seq_len]

            loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
            loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_epoch_loss)  # Adjust learning rate based on validation loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss}")


if __name__ == "__main__":
    train_file = "../data_creation/results/mix_structured_data_proposal_filtered_amazon.json"
    test_file = "amazon_test.json"
    model_path = "amazon_proposal_1"

    with open(train_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(len(data))

    def flatten_dict(d):
        """ Chuyển đổi dict thành chuỗi JSON để tránh lỗi PyArrow """
        return json.dumps(d) if isinstance(d, dict) else d

    # Chuẩn hóa dữ liệu để tránh lỗi nested dictionary
    for entry in data:
        for i, oa in enumerate(entry["input"]["oas"]):
            # Chuyển dict sentiment thành string JSON
            entry["input"]["oas"][i][2] = flatten_dict(oa[2])

        for i, is_entry in enumerate(entry["input"]["iss"]):
            # Chuyển dict sentiment thành string JSON
            entry["input"]["iss"][i]["sentiment"] = flatten_dict(is_entry["sentiment"])
    dataset = Dataset.from_list(data)

    bart_model_name = "facebook/bart-large"
    model = DualEncoderBART(bart_model_name)
    tokenizer = BartTokenizer.from_pretrained(bart_model_name)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-5, 
        betas=(0.9, 0.999), 
        eps=1e-08
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    train_model(model, dataloader, optimizer, num_epochs=10, device="cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model.save(model_path)
    print(f"Model and tokenizer saved at {model_path}")