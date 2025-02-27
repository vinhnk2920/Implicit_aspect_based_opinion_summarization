import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rouge import Rouge  # Sử dụng thư viện rouge để tính ROUGE
import numpy as np

# Set up the Hugging Face token
HUGGINGFACE_TOKEN = "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"

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load tokenizer and model with the correct token
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, token=HUGGINGFACE_TOKEN).to(device)

# Create a text-generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Load the JSON file
with open("test_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Function to convert bullet-point format to a paragraph
def convert_to_paragraph(summary_text):
    lines = summary_text.split("\n")
    sentences = []
    
    for line in lines:
        line = line.strip(" *+").replace(": ", " - ")
        if line:
            sentences.append(line)
    
    paragraph = " ".join(sentences)
    paragraph = paragraph.replace("- Opinion", "The opinions on").replace("- Implicit opinion", "Implicitly, people valued")
    
    return paragraph

# Initialize ROUGE scorer
rouge = Rouge()

# List to store all ROUGE scores
all_rouge_scores = []

# Store results
results = []

# Process each entry and generate a summary
for entry in data:
    reviews = entry["reviews"]
    ground_truth_summary = entry.get("summary", "").strip()  # Get ground truth summary

    # Construct the prompt
    prompt = (
        "Summarize the following reviews in a coherent paragraph, avoiding bullet points. "
        "Ensure the summary captures aspect-opinion pairs and implicit opinions while maintaining readability.\n\n"
    )
    for key, review in reviews.items():
        prompt += f"{key}: {review}\n"

    prompt += "\nSummary:"

    # Generate the summary
    output = generator(prompt, max_new_tokens=150, temperature=0.7, do_sample=True)

    # Extract the generated summary and format it
    generated_text = output[0]["generated_text"]
    generated_summary = generated_text.split("Summary:")[-1].strip()
    formatted_summary = convert_to_paragraph(generated_summary)

    # Compute ROUGE Score
    rouge_scores = rouge.get_scores(formatted_summary, ground_truth_summary, avg=True)

    # Extract F1 scores for rouge-1, rouge-2, rouge-l
    extracted_scores = {
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"]
    }

    all_rouge_scores.append(extracted_scores)

    # Store results
    results.append({
        "reviews": reviews,
        "ground_truth_summary": ground_truth_summary,
        "generated_summary": formatted_summary,
        "rouge_score": extracted_scores
    })

    print(f"Generated Summary:\n{formatted_summary}")
    print(f"ROUGE Score: {extracted_scores}")
    print("-" * 80)

# Compute average ROUGE Score
average_rouge = {
    "rouge-1": np.mean([score["rouge-1"] for score in all_rouge_scores]),
    "rouge-2": np.mean([score["rouge-2"] for score in all_rouge_scores]),
    "rouge-l": np.mean([score["rouge-l"] for score in all_rouge_scores])
}

# Print average ROUGE Score
print("\nAverage ROUGE Score:")
print(average_rouge)

# Save results to a JSON file
output_filename = "rouge_results.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump({"results": results, "average_rouge": average_rouge}, f, indent=4, ensure_ascii=False)

print(f"\nROUGE scores saved to {output_filename}")
