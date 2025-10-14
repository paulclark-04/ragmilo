#!/usr/bin/env python3
"""
Script de recherche vectorielle pour le pipeline RAG.
Effectue des recherches sémantiques dans l'index FAISS et affiche les résultats pertinents.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch

# Imports pour embeddings et FAISS
from sentence_transformers import SentenceTransformer
import faiss


def load_search_resources(index_dir: str) -> Tuple[faiss.Index, Dict, SentenceTransformer]:
    """
    Charge l'index FAISS, les métadonnées et le modèle d'embedding.
    
    Args:
        index_dir (str): Répertoire contenant l'index et les métadonnées
        
    Returns:
        Tuple[faiss.Index, Dict, SentenceTransformer]: (index, métadonnées, modèle)
    """
    index_path = Path(index_dir)
    
    if not index_path.exists():
        raise FileNotFoundError(f"Le répertoire d'index '{index_dir}' n'existe pas.")
    
    # Charger l'index FAISS
    faiss_file = index_path / "faiss_index.bin"
    if not faiss_file.exists():
        raise FileNotFoundError(f"Fichier d'index FAISS introuvable: {faiss_file}")
    
    print(f"🔍 Chargement de l'index FAISS: {faiss_file.name}")
    index = faiss.read_index(str(faiss_file))
    
    # Charger les métadonnées
    metadata_file = index_path / "chunks_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Fichier de métadonnées introuvable: {metadata_file}")
    
    print(f"📄 Chargement des métadonnées: {metadata_file.name}")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Charger le modèle d'embedding (même que l'indexation)
    model_name = metadata['model_info']['model_name']
    print(f"🤖 Chargement du modèle d'embedding: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Afficher les informations de l'index
    model_info = metadata['model_info']
    print(f"📊 Index chargé: {model_info['total_chunks']} chunks, {model_info['embedding_dimension']}D")
    print(f"🎯 Type de recherche: {model_info['similarity_metric']}")
    
    return index, metadata, model


def search_query(
    query: str,
    index: faiss.Index,
    metadata: Dict,
    model: SentenceTransformer,
    top_k: int = 5,
    score_threshold: Optional[float] = None
) -> List[Dict]:
    """
    Effectue une recherche vectorielle pour une question donnée.
    
    Args:
        query (str): Question à rechercher
        index (faiss.Index): Index FAISS
        metadata (Dict): Métadonnées des chunks
        model (SentenceTransformer): Modèle d'embedding
        top_k (int): Nombre de résultats à retourner
        score_threshold (Optional[float]): Seuil minimum de score
        
    Returns:
        List[Dict]: Liste des résultats avec scores et métadonnées
    """
    print(f"\n🔍 Recherche: '{query}'")
    print(f"📊 Paramètres: top_k={top_k}, threshold={score_threshold}")
    
    # Générer l'embedding de la question
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    
    # Recherche dans l'index FAISS
    scores, indices = index.search(query_embedding, top_k)
    
    # Traiter les résultats
    results = []
    chunks_data = metadata['chunks']
    
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        # Filtrer par seuil de score si spécifié
        if score_threshold is not None and score < score_threshold:
            continue
            
        # Récupérer les métadonnées du chunk
        chunk_metadata = chunks_data[idx]
        
        result = {
            'rank': i + 1,
            'score': float(score),
            'chunk_id': chunk_metadata['chunk_id'],
            'text': chunk_metadata['text'],
            'word_count': chunk_metadata['word_count'],
            'source_page': chunk_metadata['source_page'],
            'vector_index': chunk_metadata['vector_index']
        }
        results.append(result)
    
    return results


def display_results(results: List[Dict], query: str) -> None:
    """
    Affiche les résultats de recherche de manière formatée.
    
    Args:
        results (List[Dict]): Résultats de la recherche
        query (str): Question originale
    """
    if not results:
        print("❌ Aucun résultat trouvé (score trop faible ou index vide)")
        return
    
    print(f"\n📋 Résultats pour: '{query}'")
    print("=" * 80)
    
    for result in results:
        print(f"\n🏆 Rang {result['rank']} | Score: {result['score']:.4f}")
        print(f"📄 Page: {result['source_page']} | Chunk ID: {result['chunk_id']} | Mots: {result['word_count']}")
        print(f"📝 Texte: {result['text'][:200]}{'...' if len(result['text']) > 200 else ''}")
        print("-" * 80)


def interactive_mode(index: faiss.Index, metadata: Dict, model: SentenceTransformer) -> None:
    """
    Mode interactif pour poser plusieurs questions successivement.
    
    Args:
        index (faiss.Index): Index FAISS
        metadata (Dict): Métadonnées des chunks
        model (SentenceTransformer): Modèle d'embedding
    """
    print("\n🎯 Mode interactif activé")
    print("💡 Tapez vos questions (ou 'quit' pour quitter)")
    print("⚙️  Commandes spéciales:")
    print("   - 'quit' ou 'exit': quitter")
    print("   - 'help': afficher l'aide")
    print("   - 'stats': afficher les statistiques de l'index")
    print("-" * 60)
    
    while True:
        try:
            query = input("\n❓ Votre question: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Au revoir!")
                break
                
            if query.lower() == 'help':
                print("\n📖 Aide:")
                print("   - Posez vos questions en français ou en anglais")
                print("   - Exemples: 'What is machine learning?', 'Qu'est-ce que l'apprentissage automatique?'")
                print("   - 'quit' pour quitter, 'stats' pour les statistiques")
                continue
                
            if query.lower() == 'stats':
                model_info = metadata['model_info']
                print(f"\n📊 Statistiques de l'index:")
                print(f"   📦 Chunks: {model_info['total_chunks']}")
                print(f"   📏 Dimensions: {model_info['embedding_dimension']}")
                print(f"   🤖 Modèle: {model_info['model_name']}")
                print(f"   🎯 Similarité: {model_info['similarity_metric']}")
                continue
            
            # Effectuer la recherche
            results = search_query(query, index, metadata, model, top_k=5)
            display_results(results, query)
            
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")


def main():
    """Interface en ligne de commande pour le script."""
    parser = argparse.ArgumentParser(
        description="Recherche vectorielle dans l'index FAISS pour le pipeline RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Recherche simple
  python src/search1.py "What is machine learning?"
  
  # Recherche avec paramètres
  python src/search1.py "Explain neural networks" --top-k 3 --threshold 0.7
  
  # Mode interactif
  python src/search1.py --interactive
  
  # Avec répertoire d'index personnalisé
  python src/search1.py "PAC learning" --index-dir data/index
        """
    )
    
    parser.add_argument(
        'query',
        nargs='?',
        help='Question à rechercher (optionnel si --interactive)'
    )
    
    parser.add_argument(
        '--index-dir',
        default='data/index',
        help='Répertoire contenant l\'index FAISS et les métadonnées (défaut: data/index)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Nombre de résultats à afficher (défaut: 5)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        help='Seuil minimum de score pour filtrer les résultats (optionnel)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Mode interactif pour poser plusieurs questions'
    )
    
    args = parser.parse_args()
    
    try:
        # Charger les ressources
        print("🚀 Initialisation de la recherche vectorielle")
        print("=" * 60)
        index, metadata, model = load_search_resources(args.index_dir)
        
        if args.interactive:
            # Mode interactif
            interactive_mode(index, metadata, model)
        else:
            # Mode ligne de commande
            if not args.query:
                print("❌ Erreur: Veuillez fournir une question ou utiliser --interactive")
                parser.print_help()
                sys.exit(1)
            
            # Effectuer la recherche
            results = search_query(
                args.query,
                index,
                metadata,
                model,
                args.top_k,
                args.threshold
            )
            
            # Afficher les résultats
            display_results(results, args.query)
            
            # Statistiques finales
            if results:
                print(f"\n📈 {len(results)} résultats trouvés")
                print(f"🎯 Score moyen: {np.mean([r['score'] for r in results]):.4f}")
            else:
                print("\n❌ Aucun résultat trouvé")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
