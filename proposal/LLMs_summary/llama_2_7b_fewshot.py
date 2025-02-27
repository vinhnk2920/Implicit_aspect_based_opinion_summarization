import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rouge import Rouge
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

# Few-shot examples (Guidance for the model)
few_shot_examples = """
Here are examples of concise summaries based on multiple reviews:

Reviews:
1. "Woww! My order: Chicken Schwarma with hummus and pita. Absolutely clean filling. Taste delicious! Will have you craving for more."
2. "I tried to order steak kebob but they made beef kebob. Tzaziki was not as requested. Taste is okay, but not Mediterranean."
3. "Very delicious food, healthy and fresh. Love the cucumber drink!"
4. "Parsley Modern Mediterranean is wonderful. Responsive staff. The food is amazing!"
5. "The food is always fresh and leaves me full. Great value for Mediterranean cuisine."

Summary:
Fresh food, high-quality ingredients, and delicious Mediterranean flavors. The restaurant offers great value and customization options. A must-try spot for Mediterranean food lovers.

---

Reviews:
1. "Mexican food that tastes like homemade. The grilled chicken is amazing!"
2. "Simple yet delicious. Reminds me of street food in Mexico."
3. "Incredible beef and chicken, but cash only."
4. "Flavorful and authentic food at unbeatable prices."

Summary:
Authentic Mexican flavors with simple, high-quality ingredients. The grilled chicken is a must-try. Prices are low, but cash-only policy may be inconvenient.

---
"""

# Function to compute ROUGE Score
def compute_rouge(reference, generated):
    rouge = Rouge()
    scores = rouge.get_scores(generated, reference, avg=True)
    return {
        "rouge-1": scores["rouge-1"]["f"],
        "rouge-2": scores["rouge-2"]["f"],
        "rouge-l": scores["rouge-l"]["f"]
    }

# Store results
results = []
all_rouge_scores = []

# Process each entry and generate a summary
for entry in data:
    reviews = entry["reviews"]
    ground_truth_summary = entry.get("summary", "").strip()  # Get ground truth summary

    # Construct the prompt with few-shot examples
    prompt = (
        "Generate a **concise summary** for the following reviews. "
        "Focus on key aspects and opinions without unnecessary details.\n\n"
        f"{few_shot_examples}\n\n"
        "Now summarize these reviews:\n\n"
    )
    for key, review in reviews.items():
        prompt += f"{key}: {review}\n"

    prompt += "\nSummary:"

    # Generate the summary
    output = generator(prompt, max_new_tokens=150, min_length=20, temperature=0.7, do_sample=True)

    # Extract the generated summary
    generated_text = output[0]["generated_text"]
    generated_summary = generated_text.split("Summary:")[-1].strip()

    # Compute ROUGE Score
    rouge_scores = compute_rouge(ground_truth_summary, generated_summary)
    all_rouge_scores.append(rouge_scores)

    # Store results
    results.append({
        "reviews": reviews,
        "ground_truth_summary": ground_truth_summary,
        "generated_summary": generated_summary,
        "rouge_score": rouge_scores
    })

    print(f"Generated Summary:\n{generated_summary}")
    print(f"ROUGE Score: {rouge_scores}")
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
output_filename = "few_shot_rouge_results.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump({"results": results, "average_rouge": average_rouge}, f, indent=4, ensure_ascii=False)

print(f"\nROUGE scores saved to {output_filename}")
