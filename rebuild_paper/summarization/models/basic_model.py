import torch
import torch.nn as nn
import torch.nn.functional as F


class DualEncoderDecoderModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, vocab_size, max_seq_len):
        super(DualEncoderDecoderModel, self).__init__()
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Embedding(max_seq_len, d_model)
        
        # Encoders for OA and IS
        self.oa_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=num_layers
        )
        self.is_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=num_layers
        )

        # Context vectors for OA and IS
        # self.v_oa = nn.Parameter(torch.randn(d_model))
        # self.v_is = nn.Parameter(torch.randn(d_model))
        self.v_oa = nn.Parameter(torch.randn(d_model) / (d_model ** 0.5)) # scaling
        self.v_is = nn.Parameter(torch.randn(d_model) / (d_model ** 0.5)) # scaling

        
        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=num_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, oa_input=None, is_input=None, tgt_input=None, test_input=None):
        # Ensure inputs are LongTensors
        if test_input is not None:  # Test mode
            test_input = test_input.long()
            
            # Generate positional encodings
            test_pos = self.positional_encoding(
                torch.arange(test_input.size(1), device=test_input.device)
            ).unsqueeze(0)
            
            # Apply embeddings and positional encodings
            test_embedded = self.embedding(test_input) + test_pos
            
            # Pass through both encoders
            ho = self.oa_encoder(test_embedded)
            hi = self.is_encoder(test_embedded)
            
            # Aggregation (similar to training mode)
            ao = torch.mean(ho, dim=1)
            ai = torch.mean(hi, dim=1)

            # Calculate scores and lambda values
            score_oa = torch.sum(ao * self.v_oa, dim=1)  # [batch_size]
            score_is = torch.sum(ai * self.v_is, dim=1)  # [batch_size]

            # Avoid overflow with clamping
            score_oa = torch.clamp(score_oa, min=-50, max=50)
            score_is = torch.clamp(score_is, min=-50, max=50)

            # Compute lambda values
            lambda_oa = torch.exp(score_oa) / (torch.exp(score_oa) + torch.exp(score_is))
            lambda_is = 1 - lambda_oa

            # Reshape lambdas for broadcasting
            lambda_oa = lambda_oa.unsqueeze(1).unsqueeze(-1)
            lambda_is = lambda_is.unsqueeze(1).unsqueeze(-1)

            # Weighted aggregation
            co_t = lambda_oa * ho
            ci_t = lambda_is * hi
            context = torch.cat((co_t, ci_t), dim=1)  # Concatenate outputs from both encoders

        else:  # Training mode
            # Similar to the original training implementation
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
            ho = self.oa_encoder(oa_input)
            hi = self.is_encoder(is_input)

            # Aggregation
            ao = torch.mean(ho, dim=1)
            ai = torch.mean(hi, dim=1)

            score_oa = torch.sum(ao * self.v_oa, dim=1)
            score_is = torch.sum(ai * self.v_is, dim=1)

            score_oa = torch.clamp(score_oa, min=-50, max=50)
            score_is = torch.clamp(score_is, min=-50, max=50)

            # torch.exp() same as softmax
            lambda_oa = torch.exp(score_oa) / (torch.exp(score_oa) + torch.exp(score_is))
            lambda_is = 1 - lambda_oa

            lambda_oa = lambda_oa.unsqueeze(1).unsqueeze(-1)
            lambda_is = lambda_is.unsqueeze(1).unsqueeze(-1)

            co_t = lambda_oa * ho
            ci_t = lambda_is * hi

            context = torch.cat((co_t, ci_t), dim=1)  # Concatenate

        # Pass through decoder
        decoder_output = self.decoder(tgt_input, memory=context)

        # Output layer
        output = self.output_layer(decoder_output)
        return output

