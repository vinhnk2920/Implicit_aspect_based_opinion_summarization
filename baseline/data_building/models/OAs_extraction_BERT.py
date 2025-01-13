import torch
from transformers import BertTokenizer, BertForTokenClassification

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=4)  # Assume 4 labels

# Input text
input_text = "The service was slow but the pastries were fresh and delicious."

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, is_split_into_words=False)

# Dummy labels (for testing purposes only)
seq_length = inputs["input_ids"].shape[1]
labels = torch.tensor([[0] * seq_length])  # Assume all tokens are 'O'

# Forward pass
outputs = model(**inputs)
logits = outputs.logits

# Convert logits to predictions (get the highest scoring label for each token)
predictions = torch.argmax(logits, dim=-1)

# Decode the tokens and their labels
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
predicted_labels = predictions[0].tolist()

# Map label IDs to label names
id_to_label = {0: "O", 1: "B-ASPECT", 2: "I-ASPECT", 3: "B-OPINION"}

# Extract aspect-opinion pairs
aspect_opinion_pairs = []
current_aspect = None
current_opinion = None

for token, label_id in zip(tokens, predicted_labels):
    label = id_to_label[label_id]
    
    if label == "B-ASPECT":
        # Save previous aspect-opinion pair if exists
        if current_aspect and current_opinion:
            aspect_opinion_pairs.append((current_aspect, current_opinion))
        # Start a new aspect
        current_aspect = token
        current_opinion = None  # Reset current opinion

    elif label == "I-ASPECT" and current_aspect:
        # Continue the aspect
        current_aspect += f" {token}"

    elif label == "B-OPINION":
        # Start a new opinion
        current_opinion = token

    elif label == "O":
        # Save pair if the sequence ends
        if current_aspect and current_opinion:
            aspect_opinion_pairs.append((current_aspect, current_opinion))
            current_aspect = None
            current_opinion = None

# Append the last pair if applicable
if current_aspect and current_opinion:
    aspect_opinion_pairs.append((current_aspect, current_opinion))

# Print results
print("Tokens:", tokens)
print("Predicted Labels:", [id_to_label[label] for label in predicted_labels])
print("Extracted Aspect-Opinion Pairs:", aspect_opinion_pairs)