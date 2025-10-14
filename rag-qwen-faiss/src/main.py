from pathlib import Path
from embeddings import load_embedder, embed_texts
from indexing import create_faiss_index
from search import load_index, search

# --- 1. Charger le corpus ---
data_path = Path("data/exemple.txt")
lines = [line.strip() for line in data_path.read_text(encoding="utf-8").split("\n") if line.strip()]
metadatas = [{"text": l} for l in lines]

# --- 2. Créer les embeddings ---
embedder = load_embedder()
vectors = embed_texts(lines, embedder)

# --- 3. Créer l’index FAISS ---
create_faiss_index(vectors, metadatas, output_dir="data")

# --- 4. Charger et tester ---
index, metas = load_index()
query = "Comment fonctionne un réseau de neurones ?"

results = search(query, embedder, index, metas, k=2)

print("\nQuestion :", query)
for meta, score in results:
    print(f"→ {meta['text']} (score {score:.4f})")
