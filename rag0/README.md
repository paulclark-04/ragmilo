ECE Paris RAG (Local MVP)

## Vision & Cadre
- **Objectif** : répondre aux questions des étudiant·e·s ECE Paris à partir des cours PDF et des notes Milo.
- **Périmètre** : indexation par matière, enseignant, promo, semestre ; fonctionnement 100% local.
- **Principes de design** :
  1. Minimalisme (code simple, lisible)  
  2. Zéro hallucination (abstention si info manquante)  
  3. Traçabilité (citations doc/page/score)  
  4. Métadonnées au cœur (filtres matière/enseignant/promo)  
  5. Local-first (pas de dépendances cloud)
- **Contraintes Milo** : réponse en français, confidentialité, intégration future dans la plateforme Milo, exécution sur laptop étudiant.
- **Références** : tutoriel HuggingFace “Make your own RAG”, approche pure Python.

## Setup
- Python 3.10+ et Ollama avec les modèles gguf nécessaires (embedding + LLM).
- Installer les dépendances Python :
  ```bash
  pip install fastapi uvicorn pymupdf faiss-cpu rank-bm25 numpy
  ```
- Vérifier que les modèles Ollama sont disponibles hors-ligne.

## Pipeline (vue d’ensemble)
1. **Ingestion** : PDF → PyMuPDF → chunk de 500 mots (overlap 50) → embeddings Ollama → stockage JSON + indexes FAISS/BM25.
2. **Requête** : question utilisateur → embedding → retrieval hybride (vecteurs + BM25) → filtres métadonnées.
3. **Réponse** : prompt strict (citations obligatoires, abstention si seuil non atteint) → LLM via Ollama.
4. **Sortie** : JSON structuré (`answer`, `sources`, `confidence`, stats de retrieval) et UI chat.

## Ingestion
- Construire la base vectorielle et les indexes hybrides à partir d’un ou plusieurs PDF :
  ```bash
  python3 ingest_pdf.py --pdf cours_ml.pdf --pdf cours_algebre.pdf \
    --matiere "Machine Learning" --enseignant "Jean Dupont" \
    --promo 2025 --semestre S1 \
    --embed-model hf.co/CompendiumLabs/bge-base-en-v1.5-gguf \
    --output vector_db.json --append
  ```
- Sorties générées :
  - `vector_db.json` (chunks + métadonnées + embeddings)
  - `vector_index.faiss` (index FAISS normalisé)
  - `bm25_index.pkl` (index lexical BM25)
  - `index_meta.json` (métadonnées d’ingestion : modèle d’embedding, nb de chunks)
- Chaque PDF reçoit un `doc_id` lisible (ex. `algebre-cours1`), et chaque chunk un `chunk_id` `docid:page:index`.

## Querying (CLI)
- Exemple :
  ```bash
  python3 demo.py --question "Explique le perceptron" --matiere "Machine Learning" \
    --top-n 3 --threshold 0.35 --vector-k 20 --bm25-k 40 --alpha 0.65
  ```
- Le CLI affiche la question, la réponse, puis les sources (avec score combiné, score vectoriel, score BM25, doc/page/index).
- `--json` renvoie la réponse formatée pour intégration Milo.

## Web UI
- Lancer l’API + le front :
  ```bash
  uvicorn server:app --reload
  ```
- Ouvrir `http://localhost:8000` pour utiliser l’interface chat (filtres dynamiques matière/enseignant/promo/semestre, indicateur de frappe, citations cliquables).
- Endpoint REST principal : `POST /api/ask` (même payload que `demo.py`).

## Sortie JSON (structure)
- `answer` : texte généré (ou “Information non trouvée…”)
- `sources` : liste de `{doc_id, doc_label, page, chunk_index, chunk_id, fragment, score, vector_score, lexical_score, matiere, enseignant}`
- `confidence` : moyenne des scores top-k
- `metadata_used` : filtres appliqués
- `retrieval_stats` : `top1`, `avg_topk`, `threshold`, `vector_top1`, `lexical_top1`, `k`

## Notes & Bonnes pratiques
- Par défaut, l’embedding est anglophone (`bge-base-en`), recommander un modèle FR/multi (ex. `bge-m3`) si disponible en gguf.
- Structurer les PDF (titres, sections) pour de meilleurs chunks.
- Régénérer les indexes après chaque ajout/mise à jour de PDF (utiliser `--append` pour enrichir sans tout recalculer).
- Pour changer de modèle d’embedding : re-ingérer avec `--embed-model ...` et relancer `demo.py` / l’API avec le même identifiant.

## Critères et métriques (réunion Nov 7)
- Fonctionnalité : réponses correctes et sourcées.
- Qualité de réponse : précision pédagogique, citations exactes, français clair.
- Intégration Milo : compatibilité architecture cible.
- Code : lisible, documenté, maîtrisable par toute l’équipe.
- Compréhension équipe : chacun peut expliquer le pipeline.
- **KPIs visés** : Precision@3 >70 %, temps <10 s/question, hallucination 0 %, citations 100 %, filtres métadonnées 100 %.

## Limites actuelles (MVP)
- Corpus réduit (quelques dizaines de chunks).
- Chunking fixe (pas encore sémantique).
- Pas de reformulation requête ni de multi-hop.
- Dépendance à Ollama local (modèles installés manuellement).
- Validation anti-hallucination basée sur le seuil + prompt (pas encore de post-check automatique).

## Roadmap / Futurs travaux
- Renforcer la validation (post-vérification de citations, filtrage strict).
- Chunking sémantique, regroupement par chapitres, BM25+rerank avancé.
- Ingestion incrémentale à chaud (sans rebuild complet, watchers).
- Intégration Milo (API locale, SDK, instrumentation latence/logs).
- Feedback utilisateur (notations, boucle d’amélioration) et A/B testing.

## Philosophie de développement
1. Livrer le plus simple MVP fonctionnel avant d’ajouter de la complexité.
2. Basculer vers des optimisations guidées par des métriques et cas réels.
3. Gérer les risques de hallucination en priorité (seuil, consignes, validations).
4. Documenter chaque décision pour que les pairs puissent reprendre facilement.
5. Maintenir une codebase lisible pour des étudiant·e·s Junior.

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
