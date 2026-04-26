from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load a pretrained sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# A small historical complaint database
sentences = [
    "My package has not arrived yet.",
    "The delivery is delayed and tracking has not updated.",
    "I was charged twice for the same order.",
    "Customer support was rude and unhelpful."
]

# Convert all historical complaints into embeddings
embeddings = model.encode(sentences)

# New user complaint
query = "My order is late and I cannot track it."

# Convert the new complaint into an embedding
query_embedding = model.encode([query])

# Compute cosine similarity between the query and all stored complaints
scores = cosine_similarity(query_embedding, embeddings)[0]

# Get indices sorted from highest score to lowest score
top_indices = np.argsort(scores)[::-1][:2]

print("Query:")
print(query)
print("\nTop 2 most similar complaints:\n")

for rank, idx in enumerate(top_indices, start=1):
    print(f"Rank {rank}")
    print(f"Score: {scores[idx]:.4f}")
    print(f"Sentence: {sentences[idx]}\n")