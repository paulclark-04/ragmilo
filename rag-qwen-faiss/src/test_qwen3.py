from sentence_transformers import SentenceTransformer

print("⏳ Chargement du modèle Qwen3-Embedding...")
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

texts = [
    "L'intelligence artificielle permet aux machines d'apprendre.",
    "La cybersécurité protège les systèmes informatiques."
]

embeddings = model.encode(texts)
print("✅ Modèle chargé et embeddings générés.")
print("Dimension des vecteurs :", embeddings.shape)
