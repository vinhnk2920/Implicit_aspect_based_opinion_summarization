from transformers import BartTokenizer

# Load tokenizer
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["output"]
    
    # Tokenize input
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    # Tokenize output (target)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs