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
        print("Initializing simplified IS-only model...")
        super(DualEncoderBART, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained(bart_model_name)
        self.bart_is = BartForConditionalGeneration.from_pretrained(
            bart_model_name,
            dropout=0.1,
            attention_dropout=0.1
        )

    def forward(self, oas_input, iss_input, decoder_input):
        # Only use IS input
        is_encoder = self.bart_is.get_encoder()
        is_outputs = is_encoder(input_ids=iss_input["input_ids"], attention_mask=iss_input["attention_mask"])
        hi = is_outputs.last_hidden_state

        max_decoder_len = min(hi.shape[1], decoder_input.shape[1])
        decoder_input = decoder_input[:, :max_decoder_len]
        attention_mask = iss_input["attention_mask"][:, :hi.shape[1]]

        decoder_outputs = self.bart_is(
            input_ids=decoder_input,
            encoder_outputs=BaseModelOutput(last_hidden_state=hi),
            attention_mask=attention_mask
        )
        return decoder_outputs.logits

    def generate(self, oas_input, iss_input, max_length=256, min_length=100, num_beams=5):
        is_encoder = self.bart_is.get_encoder()
        is_outputs = is_encoder(input_ids=iss_input["input_ids"], attention_mask=iss_input["attention_mask"])
        hi = is_outputs.last_hidden_state

        encoder_output = BaseModelOutput(last_hidden_state=hi)
        outputs = self.bart_is.generate(
            encoder_outputs=encoder_output,
            attention_mask=iss_input["attention_mask"],
            max_length=max_length,
            min_length=min_length,
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
    iss_texts = []
    targets = []

    for item in batch:
        iss_text = " ".join([
            f"[IS] {is_entry['text']} with sentiment {json.loads(is_entry['sentiment'])['label']}"
            if isinstance(is_entry['sentiment'], str) else f"[IS] {is_entry['text']} with sentiment {is_entry['sentiment']['label']}"
            for is_entry in item["input"]["iss"]
        ])
        iss_texts.append(iss_text)
        targets.append(item["summary"])

    iss_inputs = tokenizer(iss_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    target_inputs = tokenizer(targets, return_tensors="pt", padding=True, truncation=True, max_length=512)

    return {
        "oas_input": iss_inputs,  # dummy placeholder
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

            iss_input = {key: val.to(device) for key, val in batch["iss_input"].items()}
            decoder_input = batch["decoder_input"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(None, iss_input, decoder_input)
            seq_len = outputs.shape[1]
            labels = labels[:, :seq_len]

            loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
            loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss}")


if __name__ == "__main__":
    train_file = "results/mix_structured_data_proposal_filtered_amazon.json"
    test_file = "amazon_test.json"
    model_path = "amazon_IS_only"

    with open(train_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    def flatten_dict(d):
        return json.dumps(d) if isinstance(d, dict) else d

    for entry in data:
        for i, is_entry in enumerate(entry["input"]["iss"]):
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
