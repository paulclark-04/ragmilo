ECE Paris RAG (Local MVP)

Overview
- Local, minimal RAG for ECE Paris course materials and Milo notes.
- Goals: zero hallucination, proper citations, French-first, metadata filtering.

Setup
- Requires Python 3.10+ et Ollama avec les modèles gguf nécessaires (embedding + LLM).
- Installer les dépendances Python :
  `pip install fastapi uvicorn pymupdf faiss-cpu rank-bm25 numpy`
- Vérifier que les modèles Ollama sont disponibles hors-ligne.

Ingestion
- Build the vector database and hybrid indexes from one or more PDFs:
  `python3 ingest_pdf.py --pdf cours_ml.pdf --pdf cours_algebre.pdf \
    --matiere "Machine Learning" --enseignant "Jean Dupont" \
    --promo 2025 --semestre S1 \
    --embed-model hf.co/CompendiumLabs/bge-base-en-v1.5-gguf \
    --output vector_db.json`
- Outputs:
  - `vector_db.json` (chunks + metadata + embeddings)
  - `vector_index.faiss` (FAISS cosine index)
  - `bm25_index.pkl` (BM25Okapi corpus)
  - `index_meta.json` (ingestion metadata)
- `--append` réutilise la base existante et ajoute seulement les nouveaux PDF (les `chunk_id` déjà présents sont ignorés).
- Le script assigne un `doc_id` stable par PDF et un `chunk_id` `docid:page:index`.

Querying
- Ask a question with optional metadata filters and guardrails:
  `python3 demo.py --question "Explique le perceptron" --matiere "Machine Learning" \
    --top-n 3 --threshold 0.35 --vector-k 20 --bm25-k 40 --alpha 0.65`
- `demo.py` fusionne FAISS (vecteurs) et BM25 (lexical) puis refuse de répondre si le meilleur score vectoriel est sous `--threshold`.

Web UI
- Lancer l’API + front : `uvicorn server:app --reload`
- Ouvrir `http://localhost:8000` : interface chat moderne (saisie question, filtres matière/enseignant/promo, affichage des sources et scores).
- L’API REST (POST `/api/ask`) renvoie le même JSON que `demo.py`.

Output
- CLI : question, réponse, puis sources avec scores hybrides (`score`, `vector_score`, `lexical_score`) et indice de confiance moyen.
- JSON (`--json`) : `answer`, `sources` (incl. `doc_id`, `page`, `chunk_id`, `score`, `vector_score`, `lexical_score`),
  `confidence`, `metadata_used`, `retrieval_stats` (top1, vector_top1, lexical_top1, seuil).

Notes
- Embeddings par défaut en anglais : privilégiez un modèle multilingue/FR (ex. bge-m3) disponible en GGUF.
- Gardez des PDF structurés (titres, sections) pour de meilleurs chunks.
- Regénérez les indexes après chaque ajout/mise à jour de documents.
- Pour switcher d’embedding (ex. bge-m3), passez `--embed-model` à l’ingestion puis relancez `demo.py` avec le même modèle.

Roadmap
- Mettre en place un protocole de validation manuel (cas Milo réels, revue croisée),
- Mettre en place une segmentation sémantique + regroupement par chapitres,
- Préparer un mode d’ingestion incrémentale à chaud (sans rebuild complet),
- Intégrer Milo (API locale) et instrumentation (latence, logs RAG).
