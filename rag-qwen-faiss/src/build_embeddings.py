#!/usr/bin/env python3
"""
Script de génération d'embeddings et d'indexation FAISS pour le pipeline RAG.
Génère des vecteurs sémantiques pour chaque chunk et crée un index de recherche rapide.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import torch

# Imports pour embeddings et FAISS
from sentence_transformers import SentenceTransformer
import faiss


def load_chunks(chunks_json_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Charge les chunks depuis le fichier JSON.
    
    Args:
        chunks_json_path (str): Chemin vers le fichier chunks.json
        
    Returns:
        Tuple[List[str], List[Dict]]: (textes des chunks, métadonnées des chunks)
    """
    chunks_path = Path(chunks_json_path)
    
    if not chunks_path.exists():
        raise FileNotFoundError(f"Le fichier chunks '{chunks_path}' n'existe pas.")
    
    print(f"📄 Chargement des chunks depuis: {chunks_path.name}")
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data['chunks']
    texts = [chunk['text'] for chunk in chunks]
    
    print(f"📊 {len(chunks)} chunks chargés")
    print(f"📝 Total mots: {sum(chunk['word_count'] for chunk in chunks):,}")
    
    return texts, chunks


def load_embedding_model(model_name: str = "all-mpnet-base-v2") -> SentenceTransformer:
    """
    Charge le modèle d'embedding avec gestion GPU/CPU.
    
    Args:
        model_name (str): Nom du modèle sentence-transformers
        
    Returns:
        SentenceTransformer: Modèle chargé
    """
    print(f"🤖 Chargement du modèle d'embedding: {model_name}")
    
    try:
        model = SentenceTransformer(model_name)
        
        # Afficher les informations du modèle
        device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"💻 Device utilisé: {device}")
        print(f"📏 Dimension des embeddings: {model.get_sentence_embedding_dimension()}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU disponible: {torch.cuda.get_device_name(0)}")
            print(f"💾 Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        return model
        
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du modèle {model_name}: {e}")


def generate_embeddings_batch(
    model: SentenceTransformer, 
    texts: List[str], 
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Génère les embeddings par batch avec normalisation L2.
    
    Args:
        model (SentenceTransformer): Modèle d'embedding
        texts (List[str]): Liste des textes à encoder
        batch_size (int): Taille des batches
        show_progress (bool): Afficher la barre de progression
        
    Returns:
        np.ndarray: Matrice des embeddings normalisés
    """
    print(f"🔄 Génération des embeddings (batch_size={batch_size})...")
    
    # Générer les embeddings avec sentence-transformers
    # normalize_embeddings=True pour cosine similarity
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True  # Normalisation L2 pour cosine similarity
    )
    
    print(f"✅ {len(embeddings)} embeddings générés")
    print(f"📏 Dimensions: {embeddings.shape}")
    
    return embeddings


def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Crée un index FAISS optimisé pour la recherche de similarité.
    
    Args:
        embeddings (np.ndarray): Matrice des embeddings normalisés
        
    Returns:
        faiss.Index: Index FAISS configuré
    """
    print("🔍 Création de l'index FAISS...")
    
    # Vérifier que les embeddings sont normalisés
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6):
        print("⚠️  Normalisation des embeddings...")
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Créer l'index FAISS avec Inner Product (cosine similarity)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product pour cosine similarity
    
    # Ajouter les vecteurs à l'index
    index.add(embeddings.astype('float32'))
    
    print(f"✅ Index FAISS créé avec {index.ntotal} vecteurs")
    print(f"📏 Dimension: {dimension}")
    print(f"🔍 Type d'index: IndexFlatIP (cosine similarity)")
    
    return index


def save_index_and_metadata(
    index: faiss.Index,
    chunks: List[Dict],
    embeddings: np.ndarray,
    model_name: str,
    output_dir: str
) -> None:
    """
    Sauvegarde l'index FAISS et les métadonnées.
    
    Args:
        index (faiss.Index): Index FAISS
        chunks (List[Dict]): Métadonnées des chunks
        embeddings (np.ndarray): Embeddings générés
        model_name (str): Nom du modèle utilisé
        output_dir (str): Répertoire de sortie
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder l'index FAISS
    faiss_path = output_path / "faiss_index.bin"
    faiss.write_index(index, str(faiss_path))
    print(f"💾 Index FAISS sauvegardé: {faiss_path}")
    
    # Créer les métadonnées avec mapping vector_index
    metadata = {
        "model_info": {
            "model_name": model_name,
            "embedding_dimension": embeddings.shape[1],
            "total_chunks": len(chunks),
            "index_type": "IndexFlatIP",
            "normalization": True,
            "similarity_metric": "cosine"
        },
        "chunks": []
    }
    
    # Ajouter les métadonnées des chunks avec vector_index
    for i, chunk in enumerate(chunks):
        chunk_metadata = {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],  # Truncate pour lisibilité
            "word_count": chunk["word_count"],
            "source_page": chunk["source_page"],
            "vector_index": i,  # Index du vecteur dans FAISS
            "start_position": chunk.get("start_position", 0),
            "end_position": chunk.get("end_position", 0)
        }
        metadata["chunks"].append(chunk_metadata)
    
    # Sauvegarder les métadonnées
    metadata_path = output_path / "chunks_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Métadonnées sauvegardées: {metadata_path}")
    
    # Afficher les statistiques
    print(f"\n📊 Statistiques de l'index:")
    print(f"   📦 Chunks indexés: {len(chunks)}")
    print(f"   📏 Dimension des embeddings: {embeddings.shape[1]}")
    print(f"   💾 Taille de l'index: {faiss_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"   🎯 Type de recherche: Cosine Similarity")


def build_embeddings_index(
    chunks_json_path: str,
    output_dir: str,
    model_name: str = "all-mpnet-base-v2",
    batch_size: int = 32
) -> Dict:
    """
    Fonction principale de génération d'embeddings et d'indexation.
    
    Args:
        chunks_json_path (str): Chemin vers le fichier chunks.json
        output_dir (str): Répertoire de sortie
        model_name (str): Nom du modèle d'embedding
        batch_size (int): Taille des batches
        
    Returns:
        Dict: Statistiques de l'indexation
    """
    print("🚀 Démarrage de la génération d'embeddings et d'indexation FAISS")
    print("=" * 60)
    
    # 1. Charger les chunks
    texts, chunks = load_chunks(chunks_json_path)
    
    # 2. Charger le modèle d'embedding
    model = load_embedding_model(model_name)
    
    # 3. Générer les embeddings
    embeddings = generate_embeddings_batch(model, texts, batch_size)
    
    # 4. Créer l'index FAISS
    index = create_faiss_index(embeddings)
    
    # 5. Sauvegarder l'index et les métadonnées
    save_index_and_metadata(index, chunks, embeddings, model_name, output_dir)
    
    print("\n" + "=" * 60)
    print("🎉 Indexation terminée avec succès!")
    
    return {
        "total_chunks": len(chunks),
        "embedding_dimension": embeddings.shape[1],
        "model_name": model_name,
        "index_type": "IndexFlatIP",
        "output_dir": output_dir
    }


def main():
    """Interface en ligne de commande pour le script."""
    parser = argparse.ArgumentParser(
        description="Génère des embeddings et crée un index FAISS pour le pipeline RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python src/build_embeddings.py data/chunks.json data/index
  python src/build_embeddings.py data/chunks.json data/index --model all-MiniLM-L6-v2 --batch-size 64
  python src/build_embeddings.py data/chunks.json data/index --model jinaai/jina-embeddings-v2-base-en
        """
    )
    
    parser.add_argument(
        'chunks_json_path',
        help='Chemin vers le fichier chunks.json'
    )
    
    parser.add_argument(
        'output_dir',
        help='Répertoire de sortie pour l\'index FAISS et les métadonnées'
    )
    
    parser.add_argument(
        '--model',
        default='all-mpnet-base-v2',
        help='Nom du modèle sentence-transformers (défaut: all-mpnet-base-v2)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Taille des batches pour la génération d\'embeddings (défaut: 32)'
    )
    
    args = parser.parse_args()
    
    try:
        stats = build_embeddings_index(
            args.chunks_json_path,
            args.output_dir,
            args.model,
            args.batch_size
        )
        
        print(f"\n✅ Index créé avec succès!")
        print(f"📦 {stats['total_chunks']} chunks indexés")
        print(f"🤖 Modèle: {stats['model_name']}")
        print(f"📏 Dimension: {stats['embedding_dimension']}")
        print(f"📁 Répertoire: {stats['output_dir']}")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
