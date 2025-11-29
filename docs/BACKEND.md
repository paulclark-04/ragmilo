# ECE Paris RAG Backend Documentation

## 🎯 Vision & Principes

- Répondre aux questions des étudiant·e·s à partir des PDF de cours, 100% local
- Code minimaliste, traçable, sans hallucination : cite les documents et s'abstient si l'info manque
- Métadonnées au cœur (matière, sous_matière, enseignant, promo, semestre) pour filtrer interactivement
- Fonctionne sur un laptop étudiant (Python + Ollama), documenté pour reprise rapide

## 📋 Prérequis & Installation

### Logiciels Requis
- **Python 3.10+** avec `pip`
  - macOS: `brew install python@3.10`
  - Linux: Package manager système
- **[Ollama](https://ollama.ai)** installé localement
  - macOS: `brew install --cask ollama`
  - Linux/Windows: Installeur officiel
  - Vérifier: `ollama serve`

### Modèles Ollama
```bash
# Modèle d'embedding (à lancer une fois)
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf

# Modèle de génération
ollama pull mistral:7b
```

### Dépendances Python
```bash
# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

**Dépendances principales** :
- `fastapi`, `uvicorn` - Serveurs API
- `pymupdf` - Traitement PDF
- `faiss-cpu` - Recherche vectorielle
- `rank-bm25` - Recherche par mots-clés
- `numpy` - Calculs numériques
- `ollama` - Client Ollama
- `jinja2` - Templates HTML
- `requests` - Appels HTTP

## 🏗️ Architecture

```
ragmilo/
├── backend/
│   ├── server.py              # Serveur principal RAG (port 8000)
│   ├── file_manager_rag.py    # Gestion fichiers (port 8001)
│   ├── rag_core.py            # Logique retrieval (FAISS + BM25)
│   ├── output_formatter.py    # Formatage réponses
│   ├── database_manager.py    # Gestion base SQLite
│   ├── enhanced_ingest.py     # Ingestion PDF + embeddings
│   ├── database_schema.sql    # Schéma DB
│   └── templates/
│       └── file_manager.html
│
├── data/
│   ├── rag_database.db        # Base de données SQLite
│   ├── vector_index.faiss     # Index FAISS
│   ├── bm25_index.pkl         # Index BM25
│   ├── vector_db.json         # Export JSON
│   ├── index_meta.json        # Métadonnées
│   └── uploads/               # PDF uploadés
│
├── frontend/                   # Interface utilisateur
├── docs/                      # Documentation
└── requirements.txt
```

## 🔄 Workflow

### 1. Ingestion (`enhanced_ingest.py`)
- Découpage du PDF en chunks ~500 mots (overlap 50 mots) via PyMuPDF
- Pour chaque chunk :
  1. Génération embedding BGE via Ollama
  2. Ajout métadonnées (matière, sous_matière, enseignant, semestre, promo)
  3. Stockage dans SQLite
- Export vers :
  - `data/vector_db.json` - Texte + métadonnées + embeddings
  - `data/vector_index.faiss` - Index FAISS (cosinus similarity)
  - `data/bm25_index.pkl` - Index BM25
  - `data/index_meta.json` - Configuration

### 2. Retrieval Hybride (`rag_core.HybridRetriever`)
- Requête vectorisée avec BGE (`vector_k` candidats FAISS, défaut 20)
- En parallèle, scoring BM25 (`bm25_k` candidats, défaut 40)
- Normalisation des scores (0-1)
- Fusion pondérée : `score = alpha × score_vectoriel + (1-alpha) × score_BM25`
  - `alpha = 0.65` par défaut
- Sélection des `top_n` meilleurs (défaut 3)

### 3. Filtre de Confiance
- Si `max(score_vectoriel) < threshold` (0.35 défaut)
  → Réponse : "Information non trouvée dans les sources"
- Évite les hallucinations

### 4. Génération (`server.py`)
- Chunks contextualisés : `[doc_id:page:index]`
- Envoi au LLM Mistral 7b avec instruction stricte
- Format JSON : `answer`, `sources`, `confidence`, `metadata_used`, `retrieval_stats`

## 🚀 Utilisation

### Lancement des Serveurs

**Serveur RAG Principal** (port 8000) :
```bash
uvicorn backend.server:app --reload
```
- Interface vocale : http://localhost:8000/front_voice/index_voice.html
- Interface texte : http://localhost:8000/front_text/index_text.html
- API : http://localhost:8000/docs

**Gestionnaire de Fichiers** (port 8001) :
```bash
python3 -m backend.file_manager_rag
```
- Interface : http://localhost:8001
- Upload + classification + ingestion automatique

### Ingestion de Fichiers

**Méthode 1 : Via l'interface web** (recommandé)
1. Ouvrir http://localhost:8001
2. Remplir métadonnées : matière, sous_matière, enseignant, semestre, promo
3. Upload PDF
4. Statut : `En attente` → `En traitement` → `Traité`

**Méthode 2 : En ligne de commande**
```bash
python3 -m backend.enhanced_ingest \
  --pdf data/uploads/cours_ml.pdf \
  --matiere "Machine Learning" \
  --sous_matiere "Deep Learning" \
  --enseignant "Jean Dupont" \
  --promo 2027 \
  --semestre S1 \
  --output data/vector_db.json \
  --faiss-index data/vector_index.faiss \
  --bm25-index data/bm25_index.pkl \
  --db-path data/rag_database.db
```

### Requêtes

**Via l'interface web** :
- Sélectionner filtres (matière, enseignant, etc.)
- Poser la question
- Réponse avec citations cliquables

**Via API** :
```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explique le perceptron",
    "matiere": "Machine Learning",
    "top_n": 3,
    "threshold": 0.35,
    "alpha": 0.65
  }'
```

## 📊 Format de Sortie JSON

```json
{
  "answer": "Le perceptron est...",
  "sources": [
    {
      "doc_id": "ml-cours1",
      "doc_label": "Cours ML",
      "page": 5,
      "chunk_index": 2,
      "chunk_id": "ml-cours1:5:2",
      "fragment": "...",
      "score": 0.85,
      "vector_score": 0.92,
      "lexical_score": 0.68,
      "matiere": "Machine Learning",
      "enseignant": "Jean Dupont"
    }
  ],
  "confidence": 0.82,
  "metadata_used": {
    "matiere": "Machine Learning"
  },
  "retrieval_stats": {
    "top1": 0.92,
    "avg_topk": 0.82,
    "threshold": 0.35,
    "vector_k": 20,
    "bm25_k": 40
  }
}
```

## 🔧 Configuration

### Paramètres RAG (ajustables via API)
- `top_n` (3) - Nombre de chunks retournés
- `threshold` (0.35) - Seuil minimum de confiance
- `alpha` (0.65) - Pondération vectoriel/BM25
- `vector_k` (20) - Candidats FAISS
- `bm25_k` (40) - Candidats BM25

### Modèles
- **Embedding** : `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf`(768 dims)
- **LLM** : `mistral:7b`

## ⚠️ Bonnes Pratiques

1. **PDF** : Préférer des PDF structurés (titres/sections clairs)
2. **Métadonnées** : Toujours renseigner matière + enseignant minimum
3. **Indexes** : Se régénèrent automatiquement après upload
4. **Seuil** : Ajuster `threshold` selon le corpus (0.3-0.4 recommandé)
5. **Ollama** : Toujours vérifier que `ollama serve` tourne

## 🐛 Dépannage

**"Connection refused port 8000/8001"**
→ Vérifier qu'aucun autre service n'utilise ces ports

**"No module named 'backend'"**
→ Exécuter depuis la racine du projet : `cd ragmilo`

**"No module named 'fitz'"**
→ `pip install pymupdf`

**"Ollama connection failed"**
→ `ollama serve` dans un terminal séparé

**"Index FAISS contient X vecteurs, mais Y documents"**
→ Régénérer les indexes : uploader un nouveau fichier ou relancer `enhanced_ingest.py --export-only`

## 📚 Ressources

- [Documentation base de données](DATABASE.md)
- [Structure projet](../README.md)
- [Ollama Documentation](https://ollama.ai/docs)
- [FAISS Documentation](https://faiss.ai/)
