import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartConfig

class DualEncoderModel(nn.Module):
    def __init__(self, encoder_dim=768, decoder_dim=768):
        super(DualEncoderModel, self).__init__()
        
        # Define the OA Encoder and IS Encoder
        self.oa_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=encoder_dim, nhead=8),
            num_layers=6
        )
        self.is_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=encoder_dim, nhead=8),
            num_layers=6
        )

        # Define the decoder using BART's pre-trained decoder
        bart_config = BartConfig.from_pretrained("facebook/bart-large")
        self.decoder = BartForConditionalGeneration.from_pretrained("facebook/bart-large").get_decoder()
        
        self.fc_oa = nn.Linear(encoder_dim, decoder_dim)
        self.fc_is = nn.Linear(encoder_dim, decoder_dim)

        # Attention weights for OA and IS
        self.oa_attention = nn.Linear(decoder_dim, 1)
        self.is_attention = nn.Linear(decoder_dim, 1)
        
        # Final feed-forward layer
        self.fc_out = nn.Linear(decoder_dim, bart_config.vocab_size)

    def forward(self, oa_inputs, is_inputs, decoder_input_ids):
        """
        oa_inputs: Tensor of shape (batch_size, seq_len_oa, encoder_dim)
        is_inputs: Tensor of shape (batch_size, seq_len_is, encoder_dim)
        decoder_input_ids: Tensor of shape (batch_size, seq_len_dec)
        """
        # Encode OA and IS inputs
        oa_encoded = self.oa_encoder(oa_inputs)  # (batch_size, seq_len_oa, encoder_dim)
        is_encoded = self.is_encoder(is_inputs)  # (batch_size, seq_len_is, encoder_dim)

        # Aggregate representations
        oa_rep = self.fc_oa(oa_encoded)  # (batch_size, seq_len_oa, decoder_dim)
        is_rep = self.fc_is(is_encoded)  # (batch_size, seq_len_is, decoder_dim)

        # Compute attention weights
        oa_attn_weights = F.softmax(self.oa_attention(oa_rep), dim=1)  # (batch_size, seq_len_oa, 1)
        is_attn_weights = F.softmax(self.is_attention(is_rep), dim=1)  # (batch_size, seq_len_is, 1)

        # Weighted sum of encoder outputs
        oa_context = torch.sum(oa_attn_weights * oa_rep, dim=1)  # (batch_size, decoder_dim)
        is_context = torch.sum(is_attn_weights * is_rep, dim=1)  # (batch_size, decoder_dim)

        # Combine OA and IS context vectors
        combined_context = oa_context + is_context

        # Decode combined context with decoder input
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=combined_context.unsqueeze(1),
            encoder_attention_mask=None
        )

        # Generate output probabilities
        logits = self.fc_out(decoder_outputs.last_hidden_state)

        return logits

# Example usage
if __name__ == "__main__":
    # Dummy data
    batch_size = 4
    seq_len_oa = 10
    seq_len_is = 15
    seq_len_dec = 12
    encoder_dim = 768

    oa_inputs = torch.rand(batch_size, seq_len_oa, encoder_dim)
    is_inputs = torch.rand(batch_size, seq_len_is, encoder_dim)
    decoder_input_ids = torch.randint(0, 100, (batch_size, seq_len_dec))

    # Initialize model
    model = DualEncoderModel(encoder_dim=encoder_dim)
    
    # Forward pass
    outputs = model(oa_inputs, is_inputs, decoder_input_ids)
    print(outputs.shape)  # Should match (batch_size, seq_len_dec, vocab_size)
