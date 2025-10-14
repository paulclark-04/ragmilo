from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
from pathlib import Path

# === 1. Charger le corpus ===
data_path = Path("data/cours.txt")
texts = [line.strip() for line in data_path.read_text(encoding="utf-8").split("\n") if line.strip()]
metadatas = [{"id": i, "text": t} for i, t in enumerate(texts)]

# === 2. Encoder les textes avec Qwen3 ===
print("⏳ Chargement du modèle Qwen3-Embedding...")
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

print(f"Encodage de {len(texts)} passages...")
vectors = model.encode(texts, convert_to_numpy=True)

# === 3. Créer l'index FAISS ===
dim = vectors.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(vectors)
print(f"✅ Index FAISS créé avec {index.ntotal} vecteurs.")

# === 4. Sauvegarder l'index + métadonnées ===
faiss.write_index(index, "data/cours_index.faiss")
with open("data/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadatas, f, ensure_ascii=False, indent=2)

print("💾 Index et métadonnées sauvegardés dans /data/")
