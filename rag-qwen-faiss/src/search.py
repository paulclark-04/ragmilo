import faiss
import json
import numpy as np

def load_index(path_index="data/cours_index.faiss", path_meta="data/metadata.json"):
    index = faiss.read_index(path_index)
    try:
        with open(path_meta, "r", encoding="utf-8") as f:
            metadatas = json.load(f)
    except FileNotFoundError:
        # Fallback to legacy filename used earlier in the project
        legacy_meta = path_meta.replace("metadata.json", "metadatas.json")
        with open(legacy_meta, "r", encoding="utf-8") as f:
            metadatas = json.load(f)
    return index, metadatas

def search(query, embedder, index, metadatas, k=2):
    q_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vec, k)
    results = [(metadatas[i], D[0][rank]) for rank, i in enumerate(I[0])]
    return results
