from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import csv

model_name = "microsoft/phi-2"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model.to(device)


def parse_response(response_text):
    category = ""
    summary = ""
    action = ""

    lines = response_text.split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith("Category:"):
            category = line.replace("Category:", "").strip()
        elif line.startswith("Summary:"):
            summary = line.replace("Summary:", "").strip()
        elif line.startswith("Action:"):
            action = line.replace("Action:", "").strip()

    return category, summary, action


def analyze_complaint(complaint_text):
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

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    category, summary, action = parse_response(result)

    return {
        "raw_response": result,
        "category": category,
        "summary": summary,
        "action": action
    }


if __name__ == "__main__":
    complaints = [
        """I ordered a phone last week, but it still hasn't arrived.
The tracking page hasn't updated for days and customer support is not responding.""",

        """The laptop I received has a cracked screen and the battery does not charge at all.
I want a replacement or refund.""",

        """I contacted support three times about my account problem, but nobody has replied clearly.
The responses were slow and not helpful.""",

        """I was charged twice for the same order and still have not received my refund.""",

        """My package was marked as delivered, but I never received it.""",

        """The headphones stopped working after one day and the left side has no sound."""
    ]

    results = []

    for i, complaint in enumerate(complaints, start=1):
        output = analyze_complaint(complaint)

        print(f"\n=== Complaint {i} ===\n")
        print("Complaint:")
        print(complaint)
        print("\nCategory:")
        print(output["category"])
        print("\nSummary:")
        print(output["summary"])
        print("\nAction:")
        print(output["action"])

        results.append({
            "id": i,
            "complaint": complaint,
            "category": output["category"],
            "summary": output["summary"],
            "action": output["action"],
            "raw_response": output["raw_response"]
        })

    with open("complaint_results_structured.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "complaint", "category", "summary", "action", "raw_response"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved results to complaint_results_structured.csv")