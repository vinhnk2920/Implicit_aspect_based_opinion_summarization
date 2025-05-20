import json
from datasets import Dataset
import os
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from torch.optim.lr_scheduler import CosineAnnealingLR

class DualEncoderBART(nn.Module):
    def __init__(self, bart_model_name="facebook/bart-large"):
        print("Initializing model (OAs only)...")
        super(DualEncoderBART, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained(bart_model_name)

        self.bart_oa = BartForConditionalGeneration.from_pretrained(
            bart_model_name, dropout=0.1, attention_dropout=0.1
        )

        hidden_size = self.bart_oa.config.d_model
        self.v_o = nn.Parameter(0.01 * torch.randn(hidden_size, 1, requires_grad=True))
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, oas_input, decoder_input):
        oa_encoder = self.bart_oa.get_encoder()
        oa_outputs = oa_encoder(
            input_ids=oas_input["input_ids"],
            attention_mask=oas_input["attention_mask"]
        )
        ho = oa_outputs.last_hidden_state

        exp_ao = torch.softmax(torch.matmul(ho, self.v_o) / self.temperature, dim=-1)
        ct = exp_ao * ho

        max_decoder_len = min(ct.shape[1], decoder_input.shape[1])
        decoder_input = decoder_input[:, :max_decoder_len]
        attention_mask = oas_input["attention_mask"][:, :ct.shape[1]]

        decoder_outputs = self.bart_oa(
            input_ids=decoder_input,
            encoder_outputs=BaseModelOutput(last_hidden_state=ct),
            attention_mask=attention_mask
        )
        return decoder_outputs.logits

    def generate(self, oas_input, max_length=256, min_length=100, num_beams=5):
        oa_encoder = self.bart_oa.get_encoder()
        oa_outputs = oa_encoder(
            input_ids=oas_input["input_ids"],
            attention_mask=oas_input["attention_mask"]
        )
        ho = oa_outputs.last_hidden_state

        exp_ao = torch.softmax(torch.matmul(ho, self.v_o) / self.temperature, dim=-1)
        ct = exp_ao.unsqueeze(-1) * ho

        encoder_output = BaseModelOutput(last_hidden_state=ct)
        outputs = self.bart_oa.generate(
            encoder_outputs=encoder_output,
            attention_mask=oas_input["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            early_stopping=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

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
    targets = []

    for item in batch:
        oas_text = " ".join([
            f"[OA] {aspect}: {opinion} with sentiment {json.loads(sentiment)['label']}"
            if isinstance(sentiment, str) else f"[OA] {aspect}: {opinion} with sentiment {sentiment['label']}"
            for aspect, opinion, sentiment in item["input"]["oas"]
        ])
        oas_texts.append(oas_text)
        targets.append(item["summary"])

    oas_inputs = tokenizer(oas_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    target_inputs = tokenizer(targets, return_tensors="pt", padding=True, truncation=True, max_length=512)

    return {
        "oas_input": oas_inputs,
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

            oas_input = {k: v.to(device) for k, v in batch["oas_input"].items()}
            decoder_input = batch["decoder_input"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(oas_input, decoder_input)
            labels = labels[:, :outputs.shape[1]]

            loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
            loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train_file = "../data_creation/results/mix_structured_data_proposal_filtered_amazon.json"
    model_path = "trained_amazon_model_oa_only"

    with open(train_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Loaded training samples:", len(data))

    def flatten_dict(d):
        return json.dumps(d) if isinstance(d, dict) else d

    # Clean sentiment fields
    for entry in data:
        for i, oa in enumerate(entry["input"]["oas"]):
            entry["input"]["oas"][i][2] = flatten_dict(oa[2])

    dataset = Dataset.from_list(data)

    bart_model_name = "facebook/bart-large"
    model = DualEncoderBART(bart_model_name)
    tokenizer = BartTokenizer.from_pretrained(bart_model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-08)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    train_model(model, dataloader, optimizer, num_epochs=10, device="cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model.save(model_path)
    print(f"✅ Model saved to {model_path}")
