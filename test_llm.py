from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import csv

# Name of the local language model to load
model_name = "microsoft/phi-2"

print("Loading model...")

# The tokenizer converts text into tokens that the model can read
tokenizer = AutoTokenizer.from_pretrained(model_name)

# The model generates output based on the input prompt
model = AutoModelForCausalLM.from_pretrained(model_name)

# Use GPU if available; otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Move the model to the selected device
model.to(device)


def analyze_complaint(complaint_text):
    # The prompt gives the model clear instructions:
    # 1. what categories are allowed
    # 2. what each category means
    # 3. what output format to follow
    prompt = f"""You are a customer support assistant.

Classify the complaint into exactly one of these categories:

- Delivery issue: late delivery, missing package, tracking not updated, shipping delay
- Product issue: broken item, damaged item, defective product, wrong product received
- Service issue: rude support, slow response, unclear response, unhelpful customer service
- Billing issue: refund problem, charge problem, payment issue, invoice issue, double charge

Then provide:
1. Category
2. Summary
3. Action

Use exactly this format:

Category: <one category only>
Summary: <one sentence>
Action: <one sentence>

Complaint:
{complaint_text}

Result:
"""

    # Convert the prompt into PyTorch tensors and move them to the same device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate the model's response
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,                 # Maximum number of new tokens to generate
        do_sample=False,                    # Disable sampling for more stable output
        pad_token_id=tokenizer.eos_token_id
    )

    # The output includes both the original prompt and the newly generated text
    # Keep only the newly generated portion
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    # Decode tokens back into readable text
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Remove extra whitespace at the beginning and end
    return result.strip()


if __name__ == "__main__":
    # Example complaints to test
    complaints = [
        """I ordered a phone last week, but it still hasn't arrived.
The tracking page hasn't updated for days and customer support is not responding.""",

        """The laptop I received has a cracked screen and the battery does not charge at all.
I want a replacement or refund.""",

        """I contacted support three times about my account problem, but nobody has replied clearly.
The responses were slow and not helpful."""
    ]

    # Store all results here before saving them
    results = []

    # Process each complaint one by one
    for i, complaint in enumerate(complaints, start=1):
        output = analyze_complaint(complaint)

        # Print the complaint and the model's response to the terminal
        print(f"\n=== Complaint {i} ===\n")
        print("Complaint:")
        print(complaint)
        print("\nResponse:")
        print(output)

        # Save the result as a dictionary
        results.append({
            "id": i,
            "complaint": complaint,
            "response": output
        })

    # Write all results to a CSV file
    with open("complaint_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "complaint", "response"])
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved results to complaint_results.csv")