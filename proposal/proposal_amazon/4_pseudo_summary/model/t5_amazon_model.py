import json
from datasets import Dataset
import os
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers.modeling_outputs import BaseModelOutput

tokenizer = None  # Global tokenizer for use in collate_fn

class DualEncoderT5(nn.Module):
    def __init__(self, t5_model_name="t5-large"):
        print("Initializing T5 dual encoder model...")
        super(DualEncoderT5, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

        self.t5_oa = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.t5_is = T5ForConditionalGeneration.from_pretrained(t5_model_name)

        hidden_size = self.t5_oa.config.d_model
        self.v_o = nn.Parameter(torch.randn(1, hidden_size))  # [1, H]
        self.v_i = nn.Parameter(torch.randn(1, hidden_size))
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, oas_input, iss_input, decoder_input):
        ho = self.t5_oa.encoder(input_ids=oas_input["input_ids"], attention_mask=oas_input["attention_mask"]).last_hidden_state
        hi = self.t5_is.encoder(input_ids=iss_input["input_ids"], attention_mask=iss_input["attention_mask"]).last_hidden_state

        min_seq_len = min(ho.size(1), hi.size(1))
        ho, hi = ho[:, :min_seq_len, :], hi[:, :min_seq_len, :]

        exp_ao = torch.softmax(torch.matmul(ho, self.v_o.T) / self.temperature, dim=1)
        exp_ai = torch.softmax(torch.matmul(hi, self.v_i.T) / self.temperature, dim=1)

        lambda_o = exp_ao / (exp_ao + exp_ai)
        lambda_i = 1 - lambda_o

        ct = lambda_o * ho + lambda_i * hi

        outputs = self.t5_oa(
            encoder_outputs=(ct,),
            decoder_input_ids=decoder_input,
            attention_mask=oas_input["attention_mask"][:, :ct.size(1)]
        )
        return outputs.logits

    def generate(self, oas_input, iss_input, max_length=256, min_length=100, num_beams=5):
        ho = self.t5_oa.encoder(input_ids=oas_input["input_ids"], attention_mask=oas_input["attention_mask"]).last_hidden_state
        hi = self.t5_is.encoder(input_ids=iss_input["input_ids"], attention_mask=iss_input["attention_mask"]).last_hidden_state

        min_seq_len = min(ho.size(1), hi.size(1))
        ho, hi = ho[:, :min_seq_len, :], hi[:, :min_seq_len, :]

        exp_ao = torch.softmax(torch.matmul(ho, self.v_o.T) / self.temperature, dim=1)
        exp_ai = torch.softmax(torch.matmul(hi, self.v_i.T) / self.temperature, dim=1)

        lambda_o = exp_ao / (exp_ao + exp_ai)
        lambda_i = 1 - lambda_o

        ct = lambda_o * ho + lambda_i * hi

        # ✅ Wrap in BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=ct)

        generated_ids = self.t5_oa.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=oas_input["attention_mask"][:, :ct.size(1)],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            early_stopping=True,
            do_sample=False  # ⚠ Nếu dùng beam search, bạn nên tắt sampling
        )
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def save(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, "model_state_dict.pt"))
        self.tokenizer.save_pretrained(save_directory)

    def load(self, load_directory):
        self.load_state_dict(torch.load(os.path.join(load_directory, "model_state_dict.pt")))
        self.tokenizer = T5Tokenizer.from_pretrained(load_directory)

def collate_fn(batch):
    oas_texts, iss_texts, targets = [], [], []
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
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch in progress_bar:
            optimizer.zero_grad()

            oas_input = {key: val.to(device) for key, val in batch["oas_input"].items()}
            iss_input = {key: val.to(device) for key, val in batch["iss_input"].items()}
            decoder_input = batch["decoder_input"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(oas_input, iss_input, decoder_input)
            seq_len = outputs.shape[1]
            labels = labels[:, :seq_len]

            loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
            loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}")


if __name__ == "__main__":
    train_file = "../data_creation/results/mix_structured_data_proposal_filtered_amazon.json"
    model_path = "amazon_proposal_t5"

    with open(train_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(len(data))

    def flatten_dict(d):
        return json.dumps(d) if isinstance(d, dict) else d

    for entry in data:
        for i, oa in enumerate(entry["input"]["oas"]):
            entry["input"]["oas"][i][2] = flatten_dict(oa[2])
        for i, is_entry in enumerate(entry["input"]["iss"]):
            entry["input"]["iss"][i]["sentiment"] = flatten_dict(is_entry["sentiment"])

    dataset = Dataset.from_list(data)

    t5_model_name = "t5-large"
    model = DualEncoderT5(t5_model_name)
    tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-08)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    train_model(model, dataloader, optimizer, num_epochs=10, device="cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model.save(model_path)
    print(f"Model and tokenizer saved at {model_path}")
