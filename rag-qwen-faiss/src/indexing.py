import faiss
import numpy as np
import json
from pathlib import Path

def create_faiss_index(vectors, metadatas, output_dir="data/"):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, f"{output_dir}/cours_index.faiss")

    with open(f"{output_dir}/metadatas.json", "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

    print(f"Index sauvegardé dans {output_dir}")
