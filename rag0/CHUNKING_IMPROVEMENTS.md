# 🚀 Guide d'amélioration du chunking

## 📊 Problèmes actuels

Votre système de chunking actuel est **très basique** :
- Découpe fixe par nombre de mots (500 mots, overlap 50)
- Ignore la structure du document (titres, paragraphes, sections)
- Découpe page par page sans continuité
- Peut couper au milieu d'une phrase ou d'un concept
- Pas de segmentation sémantique

## ✅ Améliorations recommandées (par ordre de priorité)

### 1. **Chunking par paragraphes** (Facile, impact élevé)

**Avantage** : Respecte la structure naturelle du texte

```python
def chunk_by_paragraphs(text: str, max_words: int = 500, min_words: int = 50) -> List[str]:
    """
    Découpe le texte en paragraphes, puis combine les petits paragraphes
    jusqu'à atteindre max_words.
    """
    # Séparer par doubles sauts de ligne (paragraphes)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
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
                chunks.append(' '.join(current_chunk))
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
                    chunks.append(' '.join(current_chunk))
                current_chunk = [para]
                current_word_count = para_word_count
    
    # Ajouter le dernier chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
```

### 2. **Chunking avec overlap intelligent** (Facile, impact moyen)

**Avantage** : Maintient le contexte entre chunks

```python
def chunk_with_smart_overlap(text: str, size: int = 500, overlap: int = 100) -> List[str]:
    """
    Découpe avec overlap, mais essaie de commencer/terminer aux limites de phrases.
    """
    sentences = re.split(r'[.!?]+\s+', text)
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        # Prendre size mots
        chunk_words = words[i:i + size]
        
        # Si on n'est pas à la fin, essayer d'ajouter jusqu'à la fin de la phrase
        if i + size < len(words):
            # Trouver la prochaine phrase complète
            chunk_text = ' '.join(chunk_words)
            last_sentence_end = max(
                chunk_text.rfind('.'),
                chunk_text.rfind('!'),
                chunk_text.rfind('?')
            )
            if last_sentence_end > len(chunk_text) * 0.7:  # Si la phrase est assez longue
                chunk_text = chunk_text[:last_sentence_end + 1]
                chunk_words = chunk_text.split()
        
        chunk = ' '.join(chunk_words)
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        
        # Avancer avec overlap, mais en essayant de commencer à une phrase
        i += size - overlap
        
        # Ajuster pour commencer au début d'une phrase si possible
        if i < len(words):
            # Chercher le début de la prochaine phrase
            text_from_i = ' '.join(words[i:])
            next_sentence_start = min(
                text_from_i.find('.') + 1,
                text_from_i.find('!') + 1,
                text_from_i.find('?') + 1
            )
            if 0 < next_sentence_start < 50:  # Si c'est proche
                i += next_sentence_start
    
    return chunks
```

### 3. **Chunking sémantique avec détection de sections** (Moyen, impact élevé)

**Avantage** : Détecte les changements de sujet et découpe en conséquence

```python
def chunk_by_sections(text: str, max_words: int = 500) -> List[str]:
    """
    Détecte les sections (titres en majuscules, numérotations) et découpe en conséquence.
    """
    # Détecter les titres (lignes courtes en majuscules ou avec numérotation)
    lines = text.split('\n')
    sections = []
    current_section = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Détecter un titre potentiel
        is_title = (
            len(line_stripped) < 100 and  # Ligne courte
            (
                line_stripped.isupper() or  # Tout en majuscules
                re.match(r'^\d+[\.\)]\s+', line_stripped) or  # Numérotation
                re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', line_stripped)  # Titre formaté
            )
        )
        
        if is_title and current_section:
            # Sauvegarder la section précédente
            sections.append('\n'.join(current_section))
            current_section = [line_stripped]
        else:
            current_section.append(line_stripped)
    
    if current_section:
        sections.append('\n'.join(current_section))
    
    # Maintenant découper chaque section en chunks
    all_chunks = []
    for section in sections:
        section_chunks = chunk_by_paragraphs(section, max_words)
        all_chunks.extend(section_chunks)
    
    return all_chunks
```

### 4. **Chunking adaptatif avec taille variable** (Avancé, impact élevé)

**Avantage** : Ajuste la taille selon le type de contenu

```python
def adaptive_chunking(text: str, base_size: int = 500) -> List[str]:
    """
    Ajuste la taille des chunks selon le contenu :
    - Plus petits pour les définitions/théorèmes
    - Plus grands pour les explications continues
    """
    # Détecter les blocs spéciaux (définitions, théorèmes, exemples)
    definition_pattern = r'(Définition|Definition|DEFINITION)[\s:]+(.+?)(?=\n\n|\n[A-Z])'
    theorem_pattern = r'(Théorème|Theorem|THEOREME|Proposition)[\s:]+(.+?)(?=\n\n|\n[A-Z])'
    
    definitions = re.finditer(definition_pattern, text, re.IGNORECASE | re.DOTALL)
    theorems = re.finditer(theorem_pattern, text, re.IGNORECASE | re.DOTALL)
    
    # Marquer les zones spéciales
    special_zones = []
    for match in list(definitions) + list(theorems):
        special_zones.append((match.start(), match.end(), 'special'))
    
    # Découper en tenant compte des zones spéciales
    chunks = []
    words = text.split()
    i = 0
    
    while i < len(words):
        # Vérifier si on est dans une zone spéciale
        current_pos = len(' '.join(words[:i]))
        in_special = any(start <= current_pos <= end for start, end, _ in special_zones)
        
        # Taille adaptative
        chunk_size = base_size // 2 if in_special else base_size
        
        chunk_words = words[i:i + chunk_size]
        chunk = ' '.join(chunk_words)
        
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        
        # Overlap plus petit pour les zones spéciales
        overlap = 25 if in_special else 100
        i += chunk_size - overlap
    
    return chunks
```

### 5. **Chunking avec préservation du contexte** (Avancé, impact très élevé)

**Avantage** : Ajoute le titre de section au début de chaque chunk

```python
def chunk_with_context(text: str, size: int = 500) -> List[str]:
    """
    Découpe le texte mais préfixe chaque chunk avec le titre de section actuel.
    Cela améliore la compréhension lors de la recherche.
    """
    lines = text.split('\n')
    chunks = []
    current_title = ""
    current_section_text = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Détecter un nouveau titre
        if len(line_stripped) < 100 and (
            line_stripped.isupper() or
            re.match(r'^\d+[\.\)]\s+', line_stripped)
        ):
            # Traiter la section précédente
            if current_section_text:
                section_chunks = chunk_by_paragraphs(
                    '\n'.join(current_section_text),
                    size - len(current_title.split()) if current_title else size
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
        section_chunks = chunk_by_paragraphs(
            '\n'.join(current_section_text),
            size - len(current_title.split()) if current_title else size
        )
        for chunk in section_chunks:
            if current_title:
                chunks.append(f"{current_title}\n\n{chunk}")
            else:
                chunks.append(chunk)
    
    return chunks
```

## 🎯 Recommandation : Implémentation progressive

### Phase 1 (Immédiat) : Chunking par paragraphes
Remplacez `chunk_text()` par `chunk_by_paragraphs()`. **Impact immédiat** sur la qualité.

### Phase 2 (Court terme) : Ajout du contexte
Utilisez `chunk_with_context()` pour préfixer les chunks avec les titres de section.

### Phase 3 (Moyen terme) : Chunking sémantique
Implémentez `chunk_by_sections()` pour une meilleure segmentation.

## 📈 Métriques à surveiller

Après chaque amélioration, mesurez :
1. **Précision@3** : % de questions où le bon chunk est dans le top 3
2. **Score moyen de similarité** : Doit augmenter
3. **Taux de "Information non trouvée"** : Doit diminuer
4. **Taille moyenne des chunks** : Doit rester cohérente

## 🔧 Code d'intégration

Voici comment intégrer dans `ingest_pdf.py` :

```python
# Remplacer la fonction chunk_text existante
def chunk_text(text, size=500, overlap=50, method='paragraphs'):
    """
    Découpe le texte selon la méthode choisie.
    
    Args:
        text: Texte à découper
        size: Taille cible (mots)
        overlap: Overlap (mots)
        method: 'simple', 'paragraphs', 'sections', 'context'
    """
    if method == 'paragraphs':
        return chunk_by_paragraphs(text, size)
    elif method == 'sections':
        return chunk_by_sections(text, size)
    elif method == 'context':
        return chunk_with_context(text, size)
    else:  # méthode originale
        words = text.split()
        chunks = []
        for i in range(0, len(words), size - overlap):
            chunk = ' '.join(words[i:i + size])
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
        return chunks
```

Puis dans `parse_pdf()` :
```python
page_chunks = chunk_text(text, size=500, method='paragraphs')  # ou 'context'
```

## 💡 Astuces supplémentaires

1. **Nettoyer le texte avant chunking** :
   - Supprimer les espaces multiples
   - Normaliser les sauts de ligne
   - Supprimer les en-têtes/pieds de page répétitifs

2. **Gérer les tableaux et formules** :
   - Les tableaux doivent rester entiers dans un chunk
   - Les formules mathématiques ne doivent pas être coupées

3. **Ajuster selon le type de document** :
   - Cours : chunks plus grands (600-800 mots)
   - Exercices : chunks plus petits (300-400 mots)
   - Résumés : chunks moyens (400-500 mots)

