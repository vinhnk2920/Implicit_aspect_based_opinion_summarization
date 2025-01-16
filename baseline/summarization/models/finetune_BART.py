import json
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# Load tokenizer và model
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Load và chuẩn bị dữ liệu
def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# Hàm rút gọn chuỗi dài
def truncate_text(text, max_sentences=5):
    sentences = text.split(". ")  # Tách các câu bằng dấu chấm
    if len(sentences) > max_sentences:
        truncated = ". ".join(sentences[:max_sentences // 2] + sentences[-max_sentences // 2:])
        return truncated
    return text

# Preprocess train data
def preprocess_train_data(data):
    inputs = []
    outputs = []
    for item in data:
        input_parts = []

        # Nếu item là opinion-aspect pairs
        if item["type"] == "oa_pairs":
            popular = ", ".join([f"{pair['aspect']}: {pair['opinion']}" for pair in item["input"].get("popular", [])])
            unpopular = ", ".join([f"{pair['aspect']}: {pair['opinion']}" for pair in item["input"].get("unpopular", [])])
            oa_text = f"Popular: {popular}. Unpopular: {unpopular}."
            input_parts.append(oa_text)

        # Nếu item là implicit sentence
        elif item["type"] == "implicit_sentence":
            implicit_text = item["input"]
            input_parts.append(f"Implicit Sentence: {implicit_text}")

        # Kết hợp các phần thành chuỗi đầu vào
        input_text = "summarize: " + " ".join(input_parts)  # Thêm tiền tố "summarize: "
        inputs.append(input_text)

        # Nếu có output (trong trường hợp tập dev/test), thêm vào danh sách outputs
        outputs.append(item.get("output", ""))  # Nếu không có output, mặc định là chuỗi rỗng

    return inputs, outputs

# Preprocess dev/test data
def preprocess_dev_test_data(data):
    inputs = []
    outputs = []
    for item in data:
        truncated_input = truncate_text(item["input"], max_sentences=6)  # Rút gọn chuỗi
        input_text = "summarize: " + truncated_input  # Thêm tiền tố "summarize: "
        inputs.append(input_text)
        outputs.append(item["output"])
    return inputs, outputs

# Tokenize dữ liệu
def tokenize_data(inputs, outputs, max_input_length=512, max_output_length=128):
    tokenized_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        tokenized_outputs = tokenizer(outputs, max_length=max_output_length, truncation=True, padding="max_length")
    tokenized_inputs["labels"] = tokenized_outputs["input_ids"]
    return tokenized_inputs

# Load dữ liệu
train_file = "../data_preparation/results/train_data.json"
dev_file = "../data_preparation/results/dev_data.json"
test_file = "../data_preparation/results/test_data.json"

train_data = load_data(train_file)
dev_data = load_data(dev_file)
test_data = load_data(test_file)

# Preprocess dữ liệu
train_inputs, train_outputs = preprocess_train_data(train_data)
dev_inputs, dev_outputs = preprocess_dev_test_data(dev_data)
test_inputs, test_outputs = preprocess_dev_test_data(test_data)

# Tokenize dữ liệu
train_tokenized = tokenize_data(train_inputs, train_outputs)
dev_tokenized = tokenize_data(dev_inputs, dev_outputs)
test_tokenized = tokenize_data(test_inputs, test_outputs)

# Convert to DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_dict(train_tokenized),
    "dev": Dataset.from_dict(dev_tokenized),
    "test": Dataset.from_dict(test_tokenized),
})

# Training arguments
training_args = TrainingArguments(
    output_dir="./bart_results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=50,
    fp16=torch.cuda.is_available(),  # Sử dụng fp16 nếu có GPU
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Save model
model.save_pretrained("./fine_tuned_bart")
tokenizer.save_pretrained("./fine_tuned_bart")

# Generate predictions for validation and test sets
def generate_predictions(model, tokenizer, inputs, references, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Đưa model lên thiết bị
    predictions = []

    for input_text, reference_text in zip(inputs, references):
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Đưa input lên thiết bị
        outputs = model.generate(
            **inputs,
            max_length=128,  # Tăng độ dài đầu ra nếu cần
            num_beams=4,  # Beam search để cải thiện chất lượng
            early_stopping=True,  # Dừng sớm nếu đã tìm thấy kết quả tốt
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append({
            "input": input_text,
            "reference": reference_text,
            "generated": generated_text
        })

    # Save predictions to file
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to {output_file}")

# Generate validation results
generate_predictions(
    model=model,
    tokenizer=tokenizer,
    inputs=dev_inputs,
    references=dev_outputs,
    output_file="validation_results.json"
)

# Generate test results
generate_predictions(
    model=model,
    tokenizer=tokenizer,
    inputs=test_inputs,
    references=test_outputs,
    output_file="test_results.json"
)