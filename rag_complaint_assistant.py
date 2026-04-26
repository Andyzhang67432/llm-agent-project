from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import csv

# Load the LLM
llm_name = "microsoft/phi-2"

print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForCausalLM.from_pretrained(llm_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
llm.to(device)

# Load the embedding model
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Small historical complaint database
historical_complaints = [
    "My package has not arrived yet.",
    "The delivery is delayed and tracking has not updated.",
    "I was charged twice for the same order.",
    "Customer support was rude and unhelpful."
]

historical_embeddings = embed_model.encode(historical_complaints)


def retrieve_similar_complaints(query, top_k=2):
    query_embedding = embed_model.encode([query])
    scores = cosine_similarity(query_embedding, historical_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved = []
    for idx in top_indices:
        retrieved.append(historical_complaints[idx])

    return retrieved


def extract_structured_lines(text):
    category = ""
    summary = ""
    action = ""

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Category:") and not category:
            category = line.replace("Category:", "").strip()
        elif line.startswith("Summary:") and not summary:
            summary = line.replace("Summary:", "").strip()
        elif line.startswith("Action:") and not action:
            action = line.replace("Action:", "").strip()

    return category, summary, action


def analyze_complaint_with_rag(complaint_text):
    retrieved_cases = retrieve_similar_complaints(complaint_text, top_k=2)
    retrieved_text = "\n".join([f"- {case}" for case in retrieved_cases])

    prompt = f"""You are a customer support assistant.

Possible categories:
- Delivery issue
- Product issue
- Service issue
- Billing issue

Similar past complaints:
{retrieved_text}

New complaint:
{complaint_text}

Example output:
Category: Delivery issue
Summary: The complaint is about a delayed package with no tracking updates.
Action: Check the shipment status and contact delivery support.

Now write the answer for the new complaint.

Category:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = llm.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    raw_result = "Category:" + raw_result
    category, summary, action = extract_structured_lines(raw_result)

    return {
        "retrieved_cases": " | ".join(retrieved_cases),
        "category": category,
        "summary": summary,
        "action": action,
        "raw_response": raw_result
    }


if __name__ == "__main__":
    complaints = [
        "My order is late and I cannot track it.",
        "The screen of the laptop I received is cracked.",
        "Support replied very slowly and their answer was not helpful.",
        "I was charged twice for one purchase.",
        "My package says delivered but I never received it."
    ]

    results = []

    for i, complaint in enumerate(complaints, start=1):
        output = analyze_complaint_with_rag(complaint)

        print(f"\n=== Complaint {i} ===\n")
        print("Complaint:")
        print(complaint)
        print("\nRetrieved cases:")
        print(output["retrieved_cases"])
        print("\nCategory:")
        print(output["category"])
        print("\nSummary:")
        print(output["summary"])
        print("\nAction:")
        print(output["action"])

        results.append({
            "id": i,
            "complaint": complaint,
            "retrieved_cases": output["retrieved_cases"],
            "category": output["category"],
            "summary": output["summary"],
            "action": output["action"],
            "raw_response": output["raw_response"]
        })

    with open("rag_complaint_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "complaint",
                "retrieved_cases",
                "category",
                "summary",
                "action",
                "raw_response"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved results to rag_complaint_results.csv")