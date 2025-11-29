# ECE Paris RAG - Database Integration

## 📋 Vue d'ensemble

Base de données SQLite locale pour gérer et classifier les fichiers par **Matière**, **Sous-Matière**, **Enseignant**, **Semestre**, et **Promo**. Solution gratuite, locale, et performante.

## 🗄️ Architecture

###  Base de Données SQLite
- ✅ **Gratuite** - Aucun coût
- ✅ **Locale** - Fonc tionne hors ligne
- ✅ **Légère** - Un seul fichier `.db`
- ✅ **Performante** - Optimisée

### 📊 Schéma

**Fichier** : `backend/config/database_schema.sql`
**Localisation** : `data/rag_database.db`

```sql
files (table principale)
├── Classification: matiere, sous_matiere, enseignant, semestre, promo
├── Métadonnées: filename, file_path, file_size, file_type
├── Traitement: status, is_processed, chunk_count, processing_date
└── Intégrité: file_hash, doc_id, doc_label

rag_chunks (chunks RAG)
├── Référence: file_id (clé étrangère → files)
├── Contenu: chunk_text, chunk_id, page_number
├── Embeddings: embedding (binaire), embedding_model  
└── Classification: matiere, sous_matiere, enseignant, semestre, promo

search_history (analytics)
├── Requêtes: query, filters, results_count
└── Performance: response_time_ms, user_feedback

system_config (configuration)
├── Paramètres: embedding_model, chunk_size, threshold
└── Métadonnées: key, value, description
```

## 🚀 Utilisation

### Initialisation

La base de données se crée automatiquement au premier lancement :

```bash
python3 -m backend.file_manager_rag
```

### Interface Web de Gestion

```bash
# Lancer l'interface
python3 -m backend.file_manager_rag

# Ouvrir http://localhost:8001
```

**Fonctionnalités** :
- ✅ Upload PDF avec classification automatique
- ✅ Filtrage multi-critères (matière, enseignant, etc.)
- ✅ Modification métadonnées
- ✅ Suppression fichiers
- ✅ Statistiques en temps réel

### API Python

```python
from backend.database_manager import DatabaseManager

# Utilisation avec context manager
with DatabaseManager() as db:
    # Ajouter un fichier
    file_id = db.add_file(
        file_path="data/uploads/cours_ml.pdf",
        matiere="Machine Learning",
        sous_matiere="Deep Learning",
        enseignant="Jean Dupont",
        semestre="S1",
        promo="2027"
    )
    
    # Rechercher des fichiers
    files = db.get_files_by_classification(
        matiere="Machine Learning",
        enseignant="Jean Dupont"
    )
    
    # Classifications disponibles
    classifications = db.get_unique_classifications()
    # Retourne: {"matiere": [...], "sous_matiere": [...], ...}
    
    # Ajouter des chunks RAG
    chunks_data = [{
        "chunk_id": "ml-cours1:5:2",
        "chunk_text": "Le perceptron...",
        "page_number": 5,
        "chunk_index": 2,
        "embedding": np.array([...]),  # numpy array
        "embedding_model": "bge-base-en-v1.5",
        "matiere": "Machine Learning",
        "sous_matiere": "Deep Learning",
        "enseignant": "Jean Dupont",
        "semestre": "S1",
        "promo": "2027"
    }]
    db.add_rag_chunks(file_id, chunks_data)
    
    # Marquer comme traité
    db.mark_file_processed(file_id, chunk_count=len(chunks_data))
```

## 🔄 Intégration avec RAG

### Ingestion Automatique

L'interface web lance automatiquement `enhanced_ingest.py` :

```bash
# Équivalent manuel
python3 -m backend.enhanced_ingest \
    --pdf data/uploads/cours_ml.pdf \
    --matiere "Machine Learning" \
    --sous_matiere "Deep Learning" \
    --enseignant "Jean Dupont" \
    --promo 2027 \
    --semestre S1 \
    --db-path data/rag_database.db \
    --output data/vector_db.json \
    --faiss-index data/vector_index.faiss \
    --bm25-index data/bm25_index.pkl
```

### Export/Import

**Export vers JSON** (compatibilité) :
```python
from backend.database_manager import export_to_vector_db

with DatabaseManager() as db:
    export_to_vector_db(db, "data/vector_db.json")
```

**Import depuis JSON** :
```bash
python3 -m backend.enhanced_ingest \
    --import-existing \
    --output data/vector_db.json \
    --db-path data/rag_database.db
```

## 📈 Workflow Type

1. **Upload** via http://localhost:8001
2. **Classification** automatique (matière, sous_matière, enseignant, etc.)
3. **Traitement** :
   - Statut : `En attente` → `En traitement` → `Traité`
   - Chunking + embeddings + indexation
4. **Recherche** via filtres dans l'interface RAG

## Maintenance

### Sauvegarde

```bash
# Copie simple
cp data/rag_database.db data/rag_database_backup_$(date +%Y%m%d).db

# Export JSON (backup universel)
python3 -c "
from backend.database_manager import DatabaseManager, export_to_vector_db
with DatabaseManager() as db:
    export_to_vector_db(db, 'backup_vector_db.json')
"
```

### Statistiques

```python
with DatabaseManager() as db:
    # Résumé fichiers
    summary = db.get_file_summary()
    print(f"Fichiers: {len(summary)}")
    print(f"Chunks totaux: {sum(f['actual_chunks'] for f in summary)}")
    
    # Classifications uniques
    classif = db.get_unique_classifications()
    print(f"Matières: {classif['matiere']}")
```

### Nettoyage

```python
with DatabaseManager() as db:
    # Supprimer un fichier (+ ses chunks)
    db.delete_file(file_id=5)
    
    # Mettre à jour métadonnées
    db.update_file_metadata(
        file_id=7,
        sous_matiere="Réseaux de Neurones",
        description="Version mise à jour"
    )
```

## 🔍 Comparaison Solutions

| Critère         | SQLite ✅       | Cloud DB ❌  |
|-----------------|-----------------|--------------|
| **Coût**        | Gratuit         | Coûteux      |
| **Setup**       | Automatique.    | Complexe     |
| **Internet**    | Non requis      | Requis       |
| **Sécurité**    | Données locales | Risque cloud |
| **Latence**     | <1ms            | >50ms        |
| **Maintenance** | Minimale        | Élevée       |

## Support

La base SQLite est **automatiquement créée** au premier lancement. Aucune configuration manuelle nécessaire.

**Emplacement** : `data/rag_database.db`  
**Schéma** : `backend/config/database_schema.sql`  
**Gestion** : `backend/database_manager.py`

Pour plus d'informations techniques, voir [BACKEND.md](BACKEND.md).
