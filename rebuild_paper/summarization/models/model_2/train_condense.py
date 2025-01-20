import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer
from transformers import get_constant_schedule_with_warmup
from tqdm import tqdm
import json


class Condense(nn.Module):
    def __init__(self, aspect_dim, sentiment_dim, input_dim, hidden_dim, vocab_size):
        super(Condense, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.aspect_dim = aspect_dim
        self.sentiment_dim = sentiment_dim

        # Layers for aspect predictions
        self.aspect_fc = nn.Linear(input_dim, hidden_dim)
        self.aspect_out = nn.Linear(hidden_dim, aspect_dim)

        # Layers for sentiment predictions
        self.sentiment_fc = nn.Linear(input_dim, hidden_dim)
        self.sentiment_out = nn.Linear(hidden_dim, sentiment_dim)

    def forward(self, tokens, mask):
        """
        Forward pass for the model.

        Args:
            tokens (torch.Tensor): Tokenized input sequences (must be LongTensor).
            mask (torch.Tensor): Mask for padding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Aspect predictions, sentiment predictions.
        """
        tokens = tokens.long()  # Ensure tokens are of type LongTensor
        hidden = self.embedding(tokens)

        # Aspect predictions
        aspect_hidden = F.relu(self.aspect_fc(hidden))
        aspect_pred = self.aspect_out(aspect_hidden)

        # Sentiment predictions
        sentiment_hidden = F.relu(self.sentiment_fc(hidden))
        sentiment_pred = self.sentiment_out(sentiment_hidden)

        return aspect_pred, sentiment_pred

    def calculate_loss(self, aspect_pred, sentiment_pred, sent_gold):
        """
        Calculate losses for the different components.

        Args:
            aspect_pred (torch.Tensor): Aspect prediction scores [batch_size, seq_len, aspect_dim].
            sentiment_pred (torch.Tensor): Sentiment prediction scores [batch_size, seq_len, sentiment_dim].
            sent_gold (torch.Tensor): Ground truth sentiment labels [batch_size].

        Returns:
            torch.Tensor: Combined loss for training.
        """
        # Reduce predictions over sequence length by taking the mean
        aspect_pred = aspect_pred.mean(dim=1)  # [batch_size, aspect_dim]
        sentiment_pred = sentiment_pred.mean(dim=1)  # [batch_size, sentiment_dim]

        # Cross-entropy loss for aspect predictions
        aspect_loss = F.cross_entropy(aspect_pred, sent_gold)

        # Cross-entropy loss for sentiment predictions
        sentiment_loss = F.cross_entropy(sentiment_pred, sent_gold)

        # Combine losses
        total_loss = aspect_loss + sentiment_loss
        return total_loss


def pad_text(batch, pad_token=0):
    """
    Pad tokenized sequences to the same length.

    Args:
        batch (List[List[int]]): List of tokenized sequences.
        pad_token (int): Token to use for padding.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Padded sequences and attention masks.
    """
    max_len = max(len(seq) for seq in batch)
    padded_batch = [seq + [pad_token] * (max_len - len(seq)) for seq in batch]
    mask = [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in batch]
    return torch.tensor(padded_batch, dtype=torch.long), torch.tensor(mask, dtype=torch.float)


def train():
    # Configurations
    aspect_dim = 100
    sentiment_dim = 5
    input_dim = 256
    hidden_dim = 256
    learning_rate = 3e-5
    batch_size = 16
    num_epoch = 10
    warmup_steps = 8000
    train_file = "../../../../data/yelp/train/yelp_train.json"
    model_file = "condense_yelp.model"

    # Check for GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    with open(train_file, "r") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    stars = [item["stars"] for item in data]

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<movie>"]})
    vocab_size = len(tokenizer)

    model = Condense(aspect_dim, sentiment_dim, input_dim, hidden_dim, vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)

    # Training loop
    print("Start training...")
    for epoch in range(num_epoch):
        model.train()

        shuffle_indices = np.random.permutation(len(texts))
        texts = [texts[idx] for idx in shuffle_indices]
        stars = [stars[idx] for idx in shuffle_indices]

        losses = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            batch_stars = stars[i:i + batch_size]

            # Tokenize and pad sequences
            batch_tokens = [tokenizer.encode(text, truncation=True, max_length=512) for text in batch_texts]
            batch_tokens, mask = pad_text(batch_tokens)

            batch_tokens = batch_tokens.to(device)
            mask = mask.to(device)

            # Convert target labels to zero-indexed
            sent_gold = torch.tensor([star - 1 for star in batch_stars], dtype=torch.long).to(device)

            # Forward pass
            aspect_pred, sentiment_pred = model(batch_tokens, mask)

            # Compute loss
            loss = model.calculate_loss(aspect_pred, sentiment_pred, sent_gold)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

        print(f"Epoch {epoch + 1}/{num_epoch}, Loss: {np.mean(losses):.4f}")

    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), model_file)
    print("Model saved.")


if __name__ == "__main__":
    train()
