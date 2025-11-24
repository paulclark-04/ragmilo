ECE Paris RAG (Local MVP)

## Vision & principes
- Répondre aux questions des étudiant·e·s à partir des PDF de cours et des notes Milo, 100 % hors-ligne.
- Code minimaliste, traçable, sans hallucination : cite les documents et s’abstient si l’info manque.
- Métadonnées au cœur (matière, enseignant, promo, semestre) pour filtrer interactivement et préparer l’intégration Milo.
- Fonctionne sur un laptop étudiant (Python pur + Ollama), documenté pour qu’un·e membre de l’équipe puisse reprendre rapidement.

## Prérequis & installation
- macOS/Linux avec `python3.10+` et `pip`. Sur macOS, installez [Homebrew](https://brew.sh) puis `brew install python@3.10` si besoin.
- [Ollama](https://ollama.ai) installé localement (`brew install --cask ollama` sur macOS, installeur officiel sur Linux/Windows). Vérifiez que le service tourne (`ollama serve`).
- Modèles nécessaires (run une seule fois) :
  ```bash
  ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
  ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
  ```
  Vous pouvez remplacer par `bge-m3`/autre modèle gguf FR tant qu’il est présent lors de l’ingestion et de l’inférence.
- Dépendances Python (vous pouvez créer un venv `python -m venv .venv && source .venv/bin/activate`) :
  ```bash
  pip install fastapi uvicorn pymupdf faiss-cpu rank-bm25 numpy ollama
  ```
  Ajoutez `python-dotenv` ou autres libs selon vos besoins front/back.

### Installation pas-à-pas
1. **Cloner** :
   ```bash
   git clone https://github.com/paulclark-04/ragmilo.git
   cd ragmilo/rag0
   ```
2. **Créer la base SQLite** (tables + vues) :
   ```bash
   python setup_database.py --setup
   ```
   - Pour migrer un `vector_db.json` existant : `python setup_database.py --migrate vector_db.json`.
3. **(Optionnel) Importer des fichiers existants** : copiez vos PDF dans `uploads/` ou utilisez l’UI `file_manager` ci-dessous.
4. **Configurer Ollama** : démarrer `ollama serve` dans un terminal séparé, vérifier `ollama list`.
5. **Tester le pipeline CLI** (après ingestion) : `python demo.py --question ...` (voir section Utilisation).

## Workflow
1. **Ingestion (`ingest_pdf.py`)**
   - Chaque PDF est découpé en chunks ~500 mots (overlap léger) via PyMuPDF.
   - Pour chaque chunk : embedding BGE (Ollama) + ajout des métadonnées (matière, enseignant, promo, semestre).
   - Stockage :
     - `vector_db.json` → texte + métadonnées + embeddings.
     - `vector_index.faiss` → index `IndexFlatIP` pour la similarité cosinus.
     - `bm25_index.pkl` → corpus BM25 (`rank_bm25.BM25Okapi`).
     - `index_meta.json` → configuration d’ingestion (modèle, nb de chunks) pour vérifier la cohérence lors du chargement.
2. **Retrieval hybride (`demo.py` / `rag_core.HybridRetriever`)**
   - Une requête est vectorisée avec le même modèle BGE (`vector_k` candidats FAISS).
   - En parallèle, BM25 score tous les chunks (`bm25_k`).
   - Les scores sont normalisés, pondérés par `alpha` (0.65 par défaut) et fusionnés ; les `top_n` chunks gagnants conservent leurs scores vectoriel/lexical.
3. **Filtre de confiance**
   - Si le meilleur score vectoriel < `threshold` (0.35 par défaut), le pipeline renvoie “Information non trouvée…” pour éviter les hallucinations tout en exposant les stats de retrieval.
4. **Génération & formatage**
   - Les chunks sélectionnés sont contextualisés (`[doc_id:page:index]`) puis envoyés au LLM Ollama avec une instruction stricte (français, citations obligatoires).
   - `output_formatter.py` produit une réponse JSON : `answer`, `sources`, `confidence`, `metadata_used`, `retrieval_stats` (top1, moyenne, seuil, tailles FAISS/BM25).

## Utilisation
### Lancement des services
- **API RAG + Front chat** :
  ```bash
  uvicorn server:app --reload
  ```
  Front accessible sur `http://localhost:8000`.
- **File Manager (upload + classification + ingestion async)** :
  ```bash
  uvicorn file_manager:app --reload --port 8001
  ```
  Permet d’uploader un PDF, le script lance automatiquement `enhanced_ingest.py` en arrière-plan (nécessite Ollama + Faiss installés).

### Ingestion
```bash
python3 ingest_pdf.py --pdf cours_ml.pdf --pdf cours_algebre.pdf \
  --matiere "Machine Learning" --enseignant "Jean Dupont" \
  --promo 2025 --semestre S1 \
  --embed-model hf.co/CompendiumLabs/bge-base-en-v1.5-gguf \
  --output vector_db.json --append
```
- `--append` évite de recalculer les chunks déjà présents (les `chunk_id` existants sont ignorés).

### Ingestion via l’interface fichier
1. Lancer `uvicorn file_manager:app --reload --port 8001`.
2. Ouvrir `http://localhost:8001` puis remplir le formulaire (matière, enseignant, promo…).
3. À l’upload, le fichier passe en statut `en traitement`, puis `traite` quand `enhanced_ingest.py` finit (chunking, embeddings, export FAISS/BM25/JSON).
4. En cas d’erreur (ex. modèle manquant), le statut est `erreur: ...` : vérifier les logs serveur.

### Requête CLI
```bash
python3 demo.py --question "Explique le perceptron" \
  --matiere "Machine Learning" --top-n 3 --threshold 0.35 \
  --vector-k 20 --bm25-k 40 --alpha 0.65 --json
```
- Renvoie soit une réponse sourcée, soit “Information non trouvée…”.
- En mode texte, les sources affichent scores hybrides + doc/page/index.

### API & Web UI
```bash
uvicorn server:app --reload
```
- Front accessible sur `http://localhost:8000` : chat + filtres dynamiques + citations cliquables.
- Endpoint principal `POST /api/ask` (mêmes champs que `demo.py`).

### Scripts utilitaires
- `python enhanced_ingest.py --help` : ingestion + export indexes à partir de la base SQLite (support des options `--import-existing`, `--export-only`).
- `python check_files.py` : opérations de maintenance des fichiers ingérés.
- `python test_server.py` / `python test_database.py` : smoke tests API + DB.
- `python update_db_columns.py` : exemple de migration (ajout colonne `sous_matiere`).

## Sortie JSON
- `answer`: texte généré ou message d’abstention.
- `sources`: `{doc_id, doc_label, page, chunk_index, chunk_id, fragment, score, vector_score, lexical_score, matiere, enseignant}`.
- `confidence`: moyenne des scores hybrides du top-n.
- `metadata_used`: filtres réellement appliqués.
- `retrieval_stats`: `top1`, `avg_topk`, `threshold`, `vector_top1`, `lexical_top1`, `vector_k`, `bm25_k`.

## Bonnes pratiques
- Préférer des PDF structurés (titres/sections) pour améliorer le chunking.
- Régénérer les index après toute mise à jour des PDF ; conserver les `doc_id` lisibles (`algebre-cours1`).
- Surveiller le seuil d’hallucination : ajuster `threshold` et `alpha` en fonction des corpus.
- Documenter chaque choix (embedding, paramètres) dans ce README ou la base (colonne `description`).
- Penser à démarrer Ollama avant tout appel aux scripts (`ollama serve`).

## Limites & suite
- Corpus réduit, chunking encore fixe (pas de segmentation sémantique).
- Pas de reformulation de requête ni de multi-hop ; validation anti-hallucination limitée au seuil + prompt.
- Roadmap courte : validation/citations renforcées, chunking sémantique, ingestion incrémentale “à chaud”, intégration Milo + instrumentation (latence/logs), boucle de feedback utilisateur.

## Philosophie dev
1. Livrer un MVP simple et traçable avant d’optimiser.
2. Prioriser la réduction d’hallucinations (seuil, consignes, validations).
3. Mesurer avant d’itérer (Precision@3 >70 %, <10 s par question, 0 % hallucination, 100 % citations/filtres).
4. Maintenir un code compréhensible par toute l’équipe.
