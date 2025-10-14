from sentence_transformers import SentenceTransformer
import faiss, json

# === 1. Charger le modèle et l'index ===
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
index = faiss.read_index("data/cours_index.faiss")

with open("data/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# === 2. Exemple de question ===
query = "Qu'est-ce que le machine learning ?"
print(f"\n🔍 Question : {query}")

# === 3. Encoder la question ===
q_vec = model.encode([query], convert_to_numpy=True)

# === 4. Chercher les passages les plus proches ===
D, I = index.search(q_vec, k=2)

print("\n📚 Résultats les plus pertinents :")
for rank, idx in enumerate(I[0]):
    print(f"{rank+1}. {metadata[idx]['text']} (score={D[0][rank]:.4f})")
