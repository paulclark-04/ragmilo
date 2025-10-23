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
- Ouvrir `http://localhost:8000` pour l’interface chat moderne (saisie question, filtres matière/enseignant/promo, affichage des sources et scores), ou `http://localhost:8000/file_manager.html` pour la gestion des fichiers.
L’interface chat conserve les filtres dynamiques (matiere, sous_matiere, enseignant, promo, semestre) et les citations avec scores.
Le gestionnaire de fichiers et le chat utilisent la même base (rag_database.db).
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
- Toujours exécuter file_manager.py avant server.py pour s’assurer que la base est accessible et cohérente.
- sous_matiere devient un filtre standard dans toutes les requêtes de recherche et d’ingestion.
- Lors de l’ajout d’un nouveau PDF, vérifier que la colonne chunk_count se met bien à jour (sinon recharger l’écran).
- enhanced_ingest.py peut être utilisé manuellement pour debug ou ingestion en ligne de commande.

Roadmap
- Mettre en place un protocole de validation manuel (cas Milo réels, revue croisée),
- Mettre en place une segmentation sémantique + regroupement par chapitres,
- Préparer un mode d’ingestion incrémentale à chaud (sans rebuild complet),
- Intégrer Milo (API locale) et instrumentation (latence, logs RAG).
- Intégration d’un champ de recherche textuelle dans le gestionnaire de fichiers. (Pas pressé et pas indispensable)
- Ajout d’un bouton “Ré-ingérer” pour retraiter un PDF déjà importé. (Non essentiel)
- Statistiques avancées sur la taille moyenne des chunks et couverture des matières (Non essentiel)



Nouveautés Base de Données & Gestion de Fichiers
Architecture étendue
La base SQLite (rag_database.db) remplace désormais les fichiers JSON/FAISS/BM25 autonomes pour centraliser toutes les données.
Deux tables principales :
- files : métadonnées des documents (matière, sous_matière, enseignant, semestre, promo, statut de traitement, nombre de chunks).
- rag_chunks : segments textuels indexés, avec embeddings, scores, et lien vers leur fichier source.
Le champ sous_matiere a été ajouté et géré comme matiere dans toutes les requêtes, filtres et jointures.
Chaque fichier traité conserve son file_id, et chaque chunk est lié par clé étrangère à ce fichier (file_id → rag_chunks.file_id).

Nouvelles fonctions de gestion
Ajout et ingestion automatique depuis l’interface web : un PDF est ajouté, découpé, vectorisé et inséré directement dans la base.

Suivi du statut de traitement :
is_processed : indique si le PDF a été vectorisé.
chunk_count : nombre total de segments extraits et stockés.
Export automatique : la base peut être exportée vers vector_db.json pour compatibilité avec les versions précédentes.
Requêtes enrichies
Les filtres par matière, sous-matière, enseignant, promo et semestre sont supportés dans les fonctions RAG.
Nouvelle méthode get_rag_chunks_by_classification() pour récupérer dynamiquement les chunks filtrés par métadonnées.
Écran de Gestion de Fichiers (nouvelle interface)
Vue d’ensemble
Accessible via http://localhost:8000/file_manager.html
Permet la gestion complète des fichiers PDF : ajout, suppression, visualisation des métadonnées, statut de traitement et nombre de chunks.

Fonctionnalités principales
Ajout de fichier :
Formulaire d’upload avec métadonnées obligatoires (matiere, sous_matiere, enseignant, promo, semestre).
L’ingestion est automatique (pas besoin d’appeler enhanced_ingest.py manuellement).
Suivi du traitement :
Les fichiers apparaissent dans la liste avec leur statut (✅ traité / ⏳ en attente).
Le compteur Chunks: indique le nombre réel de segments stockés en base.
Résumé dynamique :
L’en-tête affiche un récapitulatif : total de fichiers, nombre de chunks, fichiers traités, et valeurs uniques des métadonnées.
Filtrage par métadonnées :
La vue de résumé récupère toutes les combinaisons distinctes de matière / sous-matière / enseignant / promo / semestre.
API associées
GET /api/files : liste complète des fichiers et de leurs métadonnées.

POST /api/files/upload : ajout et ingestion d’un nouveau PDF.

GET /api/summary : statistiques globales (chunks, fichiers, classifications).

DELETE /api/files/{id} : suppression d’un document et de ses chunks.
