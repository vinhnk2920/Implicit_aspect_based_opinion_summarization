import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
import sys

class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionFusion, self).__init__()
        self.lambda_layer = nn.Linear(hidden_dim, 1)
        self.projection = nn.Linear(hidden_dim, 512)  # Thêm layer giảm hidden size về 512

    def forward(self, ho, hi):
        batch_size, seq_len_ho, hidden_dim = ho.shape
        _, seq_len_hi, _ = hi.shape
        
        max_seq_len = max(seq_len_ho, seq_len_hi)

        # Padding để đồng bộ độ dài giữa ho và hi
        if seq_len_ho < max_seq_len:
            pad_ho = torch.zeros((batch_size, max_seq_len - seq_len_ho, hidden_dim), device=ho.device)
            ho = torch.cat([ho, pad_ho], dim=1)

        if seq_len_hi < max_seq_len:
            pad_hi = torch.zeros((batch_size, max_seq_len - seq_len_hi, hidden_dim), device=hi.device)
            hi = torch.cat([hi, pad_hi], dim=1)

        # Tính attention score giữa ho và hi
        attention_scores = torch.matmul(ho, hi.transpose(-1, -2)) / (hidden_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Áp dụng attention lên hi
        attended_hi = torch.matmul(attention_weights, hi)

        # Học trọng số lambda_o (trọng số động)
        lambda_o = torch.sigmoid(self.lambda_layer(ho))  # [B, L, 1]
        lambda_i = 1 - lambda_o  # Đảm bảo tổng trọng số bằng 1

        # Kết hợp ho và attended_hi dựa trên trọng số học được
        ct = lambda_o * ho + lambda_i * attended_hi
        print(f"Before projection: {ct.shape}")

        # **Dùng Linear để giảm số chiều từ 768 → 512**
        ct = self.projection(ct)
        print(f"After projection: {ct.shape}")

        return ct


class DualEncoderBART(nn.Module):
    def __init__(self, bart_model_name="facebook/bart-base"):
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
        self.fusion_layer = AttentionFusion(hidden_size)

    def forward(self, oas_input, iss_input, decoder_input):
        device = next(self.parameters()).device  # Lấy thiết bị của mô hình

        oa_encoder = self.bart_oa.get_encoder()
        oa_outputs = oa_encoder(
            input_ids=oas_input["input_ids"].to(device), 
            attention_mask=oas_input["attention_mask"].to(device)
        )
        ho = oa_outputs.last_hidden_state.to(device)

        is_encoder = self.bart_is.get_encoder()
        is_outputs = is_encoder(
            input_ids=iss_input["input_ids"].to(device), 
            attention_mask=iss_input["attention_mask"].to(device)
        )
        hi = is_outputs.last_hidden_state.to(device)

        # Đồng bộ tất cả tensor trên cùng device
        ho, hi = ho.to(device), hi.to(device)

        # Áp dụng projection (chắc chắn projection cũng nằm trên GPU)
        ho, hi = self.projection(ho.to(device)), self.projection(hi.to(device))

        # Kết hợp bằng AttentionFusion
        ct = self.fusion_layer(ho, hi).to(device)

        # Kiểm tra lại thiết bị trước khi truyền vào decoder
        print(f"Device of ct: {ct.device}, Device of decoder_input: {decoder_input.device}")

        decoder_outputs = self.bart_oa(
            input_ids=decoder_input.to(device),
            encoder_outputs=BaseModelOutput(last_hidden_state=ct),
            attention_mask=oas_input["attention_mask"].to(device)
        )

        return decoder_outputs.logits


    def generate(self, oas_input, iss_input, max_length=256, num_beams=5):
        """
        Generate a summary using the dual encoder with AttentionFusion.
        """
        # Encode Opinion Aspects (OA)
        oa_encoder = self.bart_oa.get_encoder()
        oa_outputs = oa_encoder(input_ids=oas_input["input_ids"], attention_mask=oas_input["attention_mask"])
        ho = oa_outputs.last_hidden_state

        # Encode Implicit Sentences (IS)
        is_encoder = self.bart_is.get_encoder()
        is_outputs = is_encoder(input_ids=iss_input["input_ids"], attention_mask=iss_input["attention_mask"])
        hi = is_outputs.last_hidden_state

        # Kết hợp `ho` và `hi` bằng AttentionFusion
        ct = self.fusion_layer(ho, hi)

        encoder_output = BaseModelOutput(last_hidden_state=ct)
        outputs = self.bart_oa.generate(
            encoder_outputs=encoder_output,
            attention_mask=oas_input["attention_mask"],  
            max_length=max_length,
            num_beams=num_beams,
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