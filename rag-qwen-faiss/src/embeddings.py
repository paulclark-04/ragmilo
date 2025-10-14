from sentence_transformers import SentenceTransformer

def load_embedder():
    print("Chargement du modèle Qwen3-Embedding...")
    return SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

def embed_texts(texts, model):
    print(f"Encodage de {len(texts)} textes...")
    return model.encode(texts, convert_to_numpy=True)
