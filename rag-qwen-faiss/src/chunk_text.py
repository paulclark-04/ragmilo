#!/usr/bin/env python3
"""
Script de découpage intelligent de texte pour le pipeline RAG.
Découpe un fichier texte en chunks de 200-300 mots avec overlap de 50 mots.
Conserve les références aux pages sources et génère un fichier JSON structuré.
"""

import sys
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple


def split_into_sentences(text: str) -> List[str]:
    """
    Divise un texte en phrases en utilisant une regex simple.
    
    Args:
        text (str): Texte à diviser
        
    Returns:
        List[str]: Liste des phrases
    """
    # Regex pour diviser sur . ! ? suivis d'espaces ou fin de ligne
    sentence_pattern = r'[.!?]+\s+'
    sentences = re.split(sentence_pattern, text.strip())
    
    # Nettoyer et filtrer les phrases vides
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def count_words(text: str) -> int:
    """
    Compte le nombre de mots dans un texte.
    
    Args:
        text (str): Texte à analyser
        
    Returns:
        int: Nombre de mots
    """
    return len(text.split())


def extract_pages_from_text(file_content: str) -> List[Dict[str, any]]:
    """
    Extrait les pages du fichier texte formaté avec <PAGE N>.
    
    Args:
        file_content (str): Contenu du fichier
        
    Returns:
        List[Dict]: Liste des pages avec numéro et contenu
    """
    pages = []
    
    # Diviser le contenu par les délimiteurs de page
    page_sections = re.split(r'<PAGE (\d+)>', file_content)
    
    # Traiter les sections (alternance: contenu, numéro_page, contenu, ...)
    for i in range(1, len(page_sections), 2):
        if i + 1 < len(page_sections):
            page_num = int(page_sections[i])
            page_content = page_sections[i + 1].strip()
            
            if page_content:  # Ignorer les pages vides
                pages.append({
                    'page_number': page_num,
                    'content': page_content,
                    'word_count': count_words(page_content)
                })
    
    return pages


def create_chunks_with_overlap(pages: List[Dict], chunk_size: int = 250, overlap: int = 50) -> List[Dict]:
    """
    Crée des chunks avec overlap à partir des pages.
    
    Args:
        pages (List[Dict]): Liste des pages extraites
        chunk_size (int): Taille cible des chunks en mots
        overlap (int): Nombre de mots d'overlap entre chunks
        
    Returns:
        List[Dict]: Liste des chunks avec métadonnées
    """
    chunks = []
    chunk_id = 1
    global_word_position = 0
    
    # Traiter chaque page
    for page in pages:
        page_content = page['content']
        page_number = page['page_number']
        
        # Diviser la page en phrases
        sentences = split_into_sentences(page_content)
        
        current_chunk_sentences = []
        current_chunk_words = 0
        overlap_words = []
        
        for sentence in sentences:
            sentence_words = count_words(sentence)
            
            # Vérifier si ajouter cette phrase dépasserait la taille cible
            if current_chunk_words + sentence_words > chunk_size and current_chunk_sentences:
                # Créer le chunk actuel
                chunk_text = ' '.join(current_chunk_sentences)
                
                # Vérifier que le chunk est dans la fourchette acceptable (150-350 mots)
                if 150 <= current_chunk_words <= 350:
                    chunk = {
                        'chunk_id': chunk_id,
                        'text': chunk_text,
                        'word_count': current_chunk_words,
                        'source_page': page_number,
                        'start_position': global_word_position - current_chunk_words + 1,
                        'end_position': global_word_position,
                        'sentences_count': len(current_chunk_sentences)
                    }
                    chunks.append(chunk)
                    chunk_id += 1
                    
                    # Préparer l'overlap pour le prochain chunk
                    overlap_words = current_chunk_sentences[-2:] if len(current_chunk_sentences) >= 2 else current_chunk_sentences[-1:]
                
                # Réinitialiser pour le prochain chunk
                current_chunk_sentences = overlap_words.copy()
                current_chunk_words = count_words(' '.join(overlap_words))
            
            # Ajouter la phrase actuelle
            current_chunk_sentences.append(sentence)
            current_chunk_words += sentence_words
            global_word_position += sentence_words
        
        # Traiter le dernier chunk de la page s'il reste du contenu
        if current_chunk_sentences and current_chunk_words >= 150:
            chunk_text = ' '.join(current_chunk_sentences)
            chunk = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'word_count': current_chunk_words,
                'source_page': page_number,
                'start_position': global_word_position - current_chunk_words + 1,
                'end_position': global_word_position,
                'sentences_count': len(current_chunk_sentences)
            }
            chunks.append(chunk)
            chunk_id += 1
    
    return chunks


def chunk_text_file(txt_path: str, json_path: str, chunk_size: int = 250, overlap: int = 50) -> Dict:
    """
    Fonction principale de chunking d'un fichier texte.
    
    Args:
        txt_path (str): Chemin vers le fichier texte d'entrée
        json_path (str): Chemin vers le fichier JSON de sortie
        chunk_size (int): Taille cible des chunks en mots
        overlap (int): Nombre de mots d'overlap
        
    Returns:
        Dict: Statistiques du chunking
    """
    txt_path = Path(txt_path)
    json_path = Path(json_path)
    
    # Vérifier que le fichier d'entrée existe
    if not txt_path.exists():
        raise FileNotFoundError(f"Le fichier texte '{txt_path}' n'existe pas.")
    
    print(f"📄 Lecture du fichier: {txt_path.name}")
    
    # Lire le fichier texte
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
    except Exception as e:
        raise Exception(f"Erreur lors de la lecture du fichier: {e}")
    
    # Extraire les pages
    print("🔍 Extraction des pages...")
    pages = extract_pages_from_text(file_content)
    print(f"📊 {len(pages)} pages trouvées")
    
    # Créer les chunks
    print(f"✂️  Création des chunks (taille: {chunk_size} mots, overlap: {overlap} mots)...")
    chunks = create_chunks_with_overlap(pages, chunk_size, overlap)
    
    # Calculer les statistiques
    total_words = sum(chunk['word_count'] for chunk in chunks)
    avg_chunk_size = total_words / len(chunks) if chunks else 0
    
    # Créer la structure de sortie
    output_data = {
        'metadata': {
            'source_file': str(txt_path),
            'total_pages': len(pages),
            'total_chunks': len(chunks),
            'total_words': total_words,
            'average_chunk_size': round(avg_chunk_size, 2),
            'chunk_size_target': chunk_size,
            'overlap_size': overlap,
            'chunking_algorithm': 'sentence-based with overlap'
        },
        'chunks': chunks
    }
    
    # Sauvegarder en JSON
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Afficher les statistiques
    print(f"\n📈 Statistiques de chunking:")
    print(f"   📄 Pages traitées: {len(pages)}")
    print(f"   📦 Chunks créés: {len(chunks)}")
    print(f"   📝 Total mots: {total_words:,}")
    print(f"   📊 Taille moyenne des chunks: {avg_chunk_size:.1f} mots")
    print(f"   💾 Fichier JSON sauvegardé: {json_path}")
    
    # Afficher la distribution des tailles de chunks
    chunk_sizes = [chunk['word_count'] for chunk in chunks]
    min_size = min(chunk_sizes) if chunk_sizes else 0
    max_size = max(chunk_sizes) if chunk_sizes else 0
    
    print(f"   📏 Taille des chunks: {min_size}-{max_size} mots")
    
    return {
        'total_pages': len(pages),
        'total_chunks': len(chunks),
        'total_words': total_words,
        'average_chunk_size': avg_chunk_size,
        'min_chunk_size': min_size,
        'max_chunk_size': max_size
    }


def main():
    """Interface en ligne de commande pour le script."""
    parser = argparse.ArgumentParser(
        description="Découpe un fichier texte en chunks intelligents avec overlap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python src/chunk_text.py data/cours_texte.txt data/chunks.json
  python src/chunk_text.py data/cours_texte.txt data/chunks.json --chunk-size 300 --overlap 75
        """
    )
    
    parser.add_argument(
        'txt_path',
        help='Chemin vers le fichier texte à découper'
    )
    
    parser.add_argument(
        'json_path',
        help='Chemin vers le fichier JSON de sortie'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=250,
        help='Taille cible des chunks en mots (défaut: 250)'
    )
    
    parser.add_argument(
        '--overlap',
        type=int,
        default=50,
        help='Nombre de mots d\'overlap entre chunks (défaut: 50)'
    )
    
    args = parser.parse_args()
    
    try:
        stats = chunk_text_file(
            args.txt_path, 
            args.json_path, 
            args.chunk_size, 
            args.overlap
        )
        
        print(f"\n🎉 Chunking terminé avec succès!")
        print(f"📦 {stats['total_chunks']} chunks créés à partir de {stats['total_pages']} pages")
        print(f"📊 Taille moyenne: {stats['average_chunk_size']:.1f} mots")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
