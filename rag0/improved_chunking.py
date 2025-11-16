"""
Module de chunking amélioré pour RAG
Remplace la fonction chunk_text basique par des méthodes plus intelligentes
"""

import re
from typing import List


def chunk_by_paragraphs(text: str, max_words: int = 500, min_words: int = 50) -> List[str]:
    """
    Découpe le texte en paragraphes, puis combine les petits paragraphes
    jusqu'à atteindre max_words. Respecte la structure naturelle du document.
    
    Args:
        text: Texte à découper
        max_words: Nombre maximum de mots par chunk
        min_words: Nombre minimum de mots pour créer un chunk
    
    Returns:
        Liste de chunks de texte
    """
    # Nettoyer le texte
    text = re.sub(r'\s+', ' ', text)  # Normaliser les espaces
    text = text.strip()
    
    # Séparer par doubles sauts de ligne (paragraphes)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Si pas de paragraphes clairs, essayer par simple saut de ligne
    if len(paragraphs) == 1 and '\n' in text:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip() and len(p.strip()) > 20]
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for para in paragraphs:
        para_words = para.split()
        para_word_count = len(para_words)
        
        # Si le paragraphe seul dépasse max_words, on le découpe
        if para_word_count > max_words:
            # Sauvegarder le chunk en cours
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) >= min_words:
                    chunks.append(chunk_text)
                current_chunk = []
                current_word_count = 0
            
            # Découper le gros paragraphe
            words = para.split()
            for i in range(0, len(words), max_words):
                chunk = ' '.join(words[i:i + max_words])
                if len(chunk.split()) >= min_words:
                    chunks.append(chunk)
        else:
            # Vérifier si on peut ajouter ce paragraphe au chunk actuel
            if current_word_count + para_word_count <= max_words:
                current_chunk.append(para)
                current_word_count += para_word_count
            else:
                # Sauvegarder le chunk actuel et commencer un nouveau
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text.split()) >= min_words:
                        chunks.append(chunk_text)
                current_chunk = [para]
                current_word_count = para_word_count
    
    # Ajouter le dernier chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text.split()) >= min_words:
            chunks.append(chunk_text)
    
    return chunks if chunks else [text]  # Fallback si aucun chunk créé


def chunk_with_smart_overlap(text: str, size: int = 500, overlap: int = 100) -> List[str]:
    """
    Découpe avec overlap, mais essaie de commencer/terminer aux limites de phrases.
    Maintient mieux le contexte entre chunks.
    
    Args:
        text: Texte à découper
        size: Taille cible en mots
        overlap: Nombre de mots de recouvrement
    
    Returns:
        Liste de chunks avec overlap intelligent
    """
    # Nettoyer le texte
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Découper en phrases (approximatif)
    sentences = re.split(r'([.!?]+\s+)', text)
    # Recombiner les phrases avec leurs ponctuations
    sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '') 
                 for i in range(0, len(sentences)-1, 2)]
    if len(sentences) % 2 == 1:
        sentences.append(sentences[-1])
    
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        # Prendre size mots
        chunk_words = words[i:min(i + size, len(words))]
        chunk_text = ' '.join(chunk_words)
        
        # Si on n'est pas à la fin, essayer de terminer à une phrase
        if i + size < len(words):
            # Trouver la dernière phrase complète dans le chunk
            last_period = chunk_text.rfind('.')
            last_excl = chunk_text.rfind('!')
            last_quest = chunk_text.rfind('?')
            last_sentence_end = max(last_period, last_excl, last_quest)
            
            # Si on trouve une fin de phrase dans les 30% finaux, on coupe là
            if last_sentence_end > len(chunk_text) * 0.7:
                chunk_text = chunk_text[:last_sentence_end + 1].strip()
                chunk_words = chunk_text.split()
        
        if len(chunk_text.strip()) > 50:
            chunks.append(chunk_text)
        
        # Avancer avec overlap
        i += len(chunk_words) - overlap
        
        # Ajuster pour commencer au début d'une phrase si possible
        if i < len(words):
            remaining_text = ' '.join(words[i:])
            # Chercher le début de la prochaine phrase
            next_period = remaining_text.find('.')
            next_excl = remaining_text.find('!')
            next_quest = remaining_text.find('?')
            
            next_sentence_start = min(
                [x for x in [next_period, next_excl, next_quest] if x >= 0],
                default=-1
            )
            
            if 0 < next_sentence_start < 100:  # Si c'est proche
                i += next_sentence_start + 1
                # Avancer jusqu'au premier mot de la phrase suivante
                while i < len(words) and words[i] in ['.', '!', '?', ' ']:
                    i += 1
    
    return chunks if chunks else [text]


def chunk_by_sections(text: str, max_words: int = 500) -> List[str]:
    """
    Détecte les sections (titres en majuscules, numérotations) et découpe en conséquence.
    Idéal pour les documents structurés (cours, manuels).
    
    Args:
        text: Texte à découper
        max_words: Nombre maximum de mots par chunk
    
    Returns:
        Liste de chunks organisés par sections
    """
    lines = text.split('\n')
    sections = []
    current_section = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            if current_section:
                current_section.append('')  # Préserver les sauts de ligne
            continue
        
        # Détecter un titre potentiel
        is_title = False
        
        # Titre en majuscules (mais pas trop long)
        if len(line_stripped) < 100 and line_stripped.isupper() and len(line_stripped) > 3:
            is_title = True
        
        # Numérotation (1., 2., I., II., etc.)
        if re.match(r'^(\d+|[IVX]+)[\.\)]\s+[A-Z]', line_stripped):
            is_title = True
        
        # Titre formaté (première lettre de chaque mot en majuscule, ligne courte)
        if (len(line_stripped) < 80 and 
            re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', line_stripped) and
            len(line_stripped.split()) < 10):
            is_title = True
        
        if is_title and current_section:
            # Sauvegarder la section précédente
            sections.append('\n'.join(current_section))
            current_section = [line_stripped]
        else:
            current_section.append(line_stripped)
    
    if current_section:
        sections.append('\n'.join(current_section))
    
    # Si aucune section détectée, utiliser le texte entier
    if len(sections) == 1:
        return chunk_by_paragraphs(text, max_words)
    
    # Découper chaque section en chunks
    all_chunks = []
    for section in sections:
        section_chunks = chunk_by_paragraphs(section, max_words)
        all_chunks.extend(section_chunks)
    
    return all_chunks if all_chunks else [text]


def chunk_with_context(text: str, size: int = 500) -> List[str]:
    """
    Découpe le texte mais préfixe chaque chunk avec le titre de section actuel.
    Cela améliore la compréhension lors de la recherche vectorielle.
    
    Args:
        text: Texte à découper
        size: Taille cible en mots
    
    Returns:
        Liste de chunks avec contexte de section
    """
    lines = text.split('\n')
    chunks = []
    current_title = ""
    current_section_text = []
    
    for line in lines:
        line_stripped = line.strip()
        
        if not line_stripped:
            if current_section_text:
                current_section_text.append('')
            continue
        
        # Détecter un nouveau titre
        is_title = (
            len(line_stripped) < 100 and (
                line_stripped.isupper() or
                re.match(r'^(\d+|[IVX]+)[\.\)]\s+', line_stripped) or
                (re.match(r'^[A-Z][a-z]+', line_stripped) and len(line_stripped.split()) < 10)
            )
        )
        
        if is_title:
            # Traiter la section précédente
            if current_section_text:
                section_text = '\n'.join(current_section_text)
                # Ajuster la taille pour tenir compte du titre
                title_words = len(current_title.split()) if current_title else 0
                section_chunks = chunk_by_paragraphs(
                    section_text,
                    max_words=size - title_words if current_title else size
                )
                # Préfixer chaque chunk avec le titre
                for chunk in section_chunks:
                    if current_title:
                        chunks.append(f"{current_title}\n\n{chunk}")
                    else:
                        chunks.append(chunk)
            
            current_title = line_stripped
            current_section_text = []
        else:
            current_section_text.append(line_stripped)
    
    # Traiter la dernière section
    if current_section_text:
        section_text = '\n'.join(current_section_text)
        title_words = len(current_title.split()) if current_title else 0
        section_chunks = chunk_by_paragraphs(
            section_text,
            max_words=size - title_words if current_title else size
        )
        for chunk in section_chunks:
            if current_title:
                chunks.append(f"{current_title}\n\n{chunk}")
            else:
                chunks.append(chunk)
    
    return chunks if chunks else [text]


def improved_chunk_text(text: str, size: int = 500, overlap: int = 50, method: str = 'paragraphs') -> List[str]:
    """
    Fonction principale de chunking amélioré.
    Remplace la fonction chunk_text basique.
    
    Args:
        text: Texte à découper
        size: Taille cible en mots
        overlap: Overlap en mots (utilisé pour certaines méthodes)
        method: Méthode de chunking
            - 'simple': Méthode originale (découpe fixe)
            - 'paragraphs': Découpe par paragraphes (RECOMMANDÉ)
            - 'sections': Découpe par sections avec détection de titres
            - 'context': Découpe avec contexte de section (MEILLEUR)
            - 'smart': Overlap intelligent avec respect des phrases
    
    Returns:
        Liste de chunks de texte
    """
    if not text or not text.strip():
        return []
    
    if method == 'paragraphs':
        return chunk_by_paragraphs(text, size)
    elif method == 'sections':
        return chunk_by_sections(text, size)
    elif method == 'context':
        return chunk_with_context(text, size)
    elif method == 'smart':
        return chunk_with_smart_overlap(text, size, overlap)
    else:  # méthode originale 'simple'
        words = text.split()
        chunks = []
        for i in range(0, len(words), size - overlap):
            chunk = ' '.join(words[i:i + size])
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
        return chunks if chunks else [text]


# Exemple d'utilisation
if __name__ == '__main__':
    sample_text = """
    CHAPITRE 1 : INTRODUCTION
    
    Les matrices sont des objets mathématiques fondamentaux. Elles permettent de représenter
    des transformations linéaires et de résoudre des systèmes d'équations.
    
    DÉFINITION
    
    Une matrice de taille m×n est un tableau rectangulaire de nombres disposés en m lignes
    et n colonnes. On note généralement une matrice par une lettre majuscule, par exemple A.
    
    EXEMPLE
    
    La matrice suivante est une matrice 2×3 :
    [1 2 3]
    [4 5 6]
    
    CHAPITRE 2 : OPÉRATIONS
    
    On peut additionner deux matrices de même taille en additionnant leurs éléments
    correspondants.
    """
    
    print("=== Méthode 'paragraphs' ===")
    chunks = improved_chunk_text(sample_text, method='paragraphs')
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk.split())} mots):\n{chunk[:200]}...")
    
    print("\n\n=== Méthode 'context' ===")
    chunks = improved_chunk_text(sample_text, method='context')
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk.split())} mots):\n{chunk[:200]}...")

