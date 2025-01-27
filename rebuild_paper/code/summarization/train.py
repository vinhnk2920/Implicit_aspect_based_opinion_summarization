import json
from datasets import Dataset
import os
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration


class DualEncoderBART(nn.Module):
    def __init__(self, bart_model_name="facebook/bart-large"):
        print("Initializing DualEncoderBART model...")
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
        self.v_o = nn.Parameter(torch.randn(hidden_size))  # Context vector for OA
        self.v_i = nn.Parameter(torch.randn(hidden_size))  # Context vector for IS

        self.temperature = nn.Parameter(torch.ones(1))  # Temperature for weighting

    def forward(self, oas_input, iss_input, decoder_input):
        oa_encoder = self.bart_oa.get_encoder()
        oa_outputs = oa_encoder(input_ids=oas_input["input_ids"], attention_mask=oas_input["attention_mask"])
        ho = oa_outputs.last_hidden_state

        is_encoder = self.bart_is.get_encoder()
        is_outputs = is_encoder(input_ids=iss_input["input_ids"], attention_mask=iss_input["attention_mask"])
        hi = is_outputs.last_hidden_state

        # Combine Context Vectors
        min_seq_len = min(ho.size(1), hi.size(1))
        ho, hi = ho[:, :min_seq_len, :], hi[:, :min_seq_len, :]
        exp_ao = torch.exp(torch.matmul(ho, self.v_o) / self.temperature)
        exp_ai = torch.exp(torch.matmul(hi, self.v_i) / self.temperature)
        lambda_o = exp_ao / (exp_ao + exp_ai)
        lambda_i = 1 - lambda_o
        ct = lambda_o.unsqueeze(-1) * ho + lambda_i.unsqueeze(-1) * hi

        decoder_outputs = self.bart_oa(
            input_ids=decoder_input,
            encoder_outputs=(ct,),
            attention_mask=oas_input["attention_mask"]
        )

        return decoder_outputs.logits

    def generate(self, oas_input, iss_input, max_length=50, num_beams=5):
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
        exp_ao = torch.exp(torch.matmul(ho, self.v_o) / self.temperature)
        exp_ai = torch.exp(torch.matmul(hi, self.v_i) / self.temperature)
        lambda_o = exp_ao / (exp_ao + exp_ai)
        lambda_i = 1 - lambda_o
        ct = lambda_o.unsqueeze(-1) * ho + lambda_i.unsqueeze(-1) * hi

        # Generate summary
        outputs = self.bart_oa.generate(
            inputs_embeds=ct,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        # Decode the generated summary
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
        oas_text = " ".join([f"[OA] {aspect}: {opinion}" for aspect, opinion in item["input"]["oas"]])
        iss_text = " ".join([f"[IS] {sentence}" for sentence in item["input"]["iss"]])
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
    print(device)
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            # Move inputs to device
            oas_input = {key: val.to(device) for key, val in batch["oas_input"].items()}
            iss_input = {key: val.to(device) for key, val in batch["iss_input"].items()}
            decoder_input = batch["decoder_input"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(oas_input, iss_input, decoder_input)

            loss_fn = nn.CrossEntropyLoss(
                ignore_index=tokenizer.pad_token_id, 
                label_smoothing=0.1
            )
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")


if __name__ == "__main__":
    train_file = "../data_creation/results/sampling/small_mix_structured_data.json"
    test_file = "test_data.json"
    model_path = "trained_dual_encoder_bart"
    output_file = "predicted_results"

    with open(train_file, "r", encoding="utf-8") as f:
        data = json.load(f)
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