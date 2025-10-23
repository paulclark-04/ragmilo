# ECE Paris RAG - Database Integration

## Vue d'ensemble

Cette solution ajoute une base de données SQLite à votre système RAG existant pour gérer et classifier les fichiers par **Matière**, **Enseignant**, **Semestre**, et **Promo**. La solution est entièrement gratuite et fonctionne localement.

## Architecture de la Solution

### 🗄️ Base de Données SQLite
- **Gratuite** : Aucun coût, aucune dépendance externe
- **Locale** : Fonctionne hors ligne, parfait pour les étudiants
- **Légère** : Un seul fichier `.db`, facile à partager
- **Performante** : Optimisée pour votre cas d'usage

### 📊 Schéma de Base de Données

```
files (table principale)
├── Classification: matiere, enseignant, semestre, promo
├── Métadonnées: filename, file_path, file_size, file_type
├── Traitement: is_processed, chunk_count, processing_date
└── Intégrité: file_hash, doc_id, doc_label

rag_chunks (chunks RAG)
├── Référence: file_id (clé étrangère)
├── Contenu: chunk_text, chunk_id, page_number
├── Embeddings: embedding (binaire), embedding_model
└── Classification: matiere, enseignant, semestre, promo

search_history (analytics)
├── Requêtes: query, filters, results_count
└── Performance: response_time_ms, user_feedback
```

## 🚀 Installation et Configuration

### 1. Prérequis
```bash
# Dépendances Python supplémentaires
pip install sqlalchemy jinja2
```

### 2. Initialisation de la Base de Données
```bash
# La base de données se crée automatiquement au premier usage
python database_manager.py
```

### 3. Migration des Données Existantes
```bash
# Importer votre vector_db.json existant
python enhanced_ingest.py --import-existing --output vector_db.json
```

## 📁 Gestion des Fichiers

### Interface Web de Gestion
```bash
# Lancer l'interface de gestion des fichiers
python file_manager.py
# Ouvrir http://localhost:8001
```

**Fonctionnalités :**
- ✅ Upload de fichiers PDF avec classification
- ✅ Filtrage par matière, enseignant, semestre, promo
- ✅ Modification des métadonnées
- ✅ Suppression de fichiers
- ✅ Statistiques et analytics

### API REST
```python
# Exemples d'utilisation
from database_manager import DatabaseManager

with DatabaseManager() as db:
    # Ajouter un fichier
    file_id = db.add_file(
        file_path="cours_ml.pdf",
        matiere="Machine Learning",
        enseignant="Jean Dupont",
        semestre="S1",
        promo="2025"
    )
    
    # Rechercher des fichiers
    files = db.get_files_by_classification(
        matiere="Machine Learning",
        enseignant="Jean Dupont"
    )
    
    # Obtenir les classifications disponibles
    classifications = db.get_unique_classifications()
```

## 🔄 Intégration avec le RAG Existant

### 1. Ingestion Améliorée
```bash
# Utiliser le nouveau script d'ingestion avec base de données
python enhanced_ingest.py \
    --pdf cours_ml.pdf \
    --matiere "Machine Learning" \
    --enseignant "Jean Dupont" \
    --promo 2025 \
    --semestre S1
```

**Avantages :**
- ✅ Stockage structuré des métadonnées
- ✅ Détection automatique des doublons
- ✅ Traçabilité complète des fichiers
- ✅ Export compatible avec l'ancien système

### 2. Export pour Compatibilité
```python
# Exporter vers le format vector_db.json existant
from database_manager import export_to_vector_db

with DatabaseManager() as db:
    export_to_vector_db(db, "vector_db.json")
```

### 3. Serveur RAG Intégré
```python
# Le serveur RAG existant peut utiliser la base de données
# pour des filtres plus avancés et une meilleure traçabilité
```

## 📈 Avantages de cette Solution

### ✅ **Gratuite et Locale**
- Aucun coût, aucune dépendance cloud
- Fonctionne hors ligne
- Données restent sur votre machine

### ✅ **Gestion Avancée**
- Classification automatique des fichiers
- Détection des doublons par hash
- Historique des modifications
- Analytics et statistiques

### ✅ **Compatibilité Totale**
- Fonctionne avec votre système RAG existant
- Export vers `vector_db.json`
- Pas de modification du code RAG principal

### ✅ **Interface Utilisateur**
- Interface web intuitive
- Upload et gestion des fichiers
- Filtrage et recherche avancés
- Statistiques en temps réel

## 🔧 Utilisation Pratique

### Workflow Typique

1. **Upload de Fichiers**
   ```bash
   # Via l'interface web ou API
   # Classification automatique par matière/enseignant/semestre/promo
   ```

2. **Traitement RAG**
   ```bash
   # Le système traite automatiquement les nouveaux fichiers
   python enhanced_ingest.py --pdf nouveau_cours.pdf --matiere "Algèbre" --enseignant "Marie Martin" --promo 2025 --semestre S1
   ```

3. **Recherche et Filtrage**
   ```python
   # Recherche avec filtres avancés
   chunks = db.get_rag_chunks_by_classification(
       matiere="Machine Learning",
       semestre="S1"
   )
   ```

4. **Analytics**
   ```python
   # Statistiques d'utilisation
   summary = db.get_file_summary()
   classifications = db.get_unique_classifications()
   ```

## 🛠️ Maintenance et Sauvegarde

### Sauvegarde
```bash
# Sauvegarder la base de données
cp rag_database.db rag_database_backup.db

# Exporter vers JSON
python -c "from database_manager import export_to_vector_db; export_to_vector_db(DatabaseManager(), 'backup.json')"
```

### Migration
```bash
# Migrer vers un nouveau système
# 1. Exporter les données
# 2. Copier le fichier .db
# 3. Réimporter sur le nouveau système
```

## 🔍 Comparaison avec les Solutions Cloud

| Critère | SQLite (Recommandé) | Cloud Database |
|---------|-------------------|----------------|
| **Coût** | ✅ Gratuit | ❌ Coûteux |
| **Complexité** | ✅ Simple | ❌ Complexe |
| **Dépendance** | ✅ Aucune | ❌ Internet requis |
| **Sécurité** | ✅ Données locales | ❌ Risque cloud |
| **Performance** | ✅ Rapide | ⚠️ Latence réseau |
| **Maintenance** | ✅ Minimale | ❌ Élevée |

## 🚀 Prochaines Étapes

1. **Test de la Solution**
   ```bash
   # Tester avec vos fichiers existants
   python enhanced_ingest.py --import-existing
   ```

2. **Interface Web**
   ```bash
   # Lancer l'interface de gestion
   python file_manager.py
   ```

3. **Intégration Graduelle**
   - Commencer avec quelques fichiers
   - Tester les fonctionnalités
   - Migrer progressivement

## 📞 Support et Questions

Cette solution est conçue pour être :
- **Simple** : Code lisible et documenté
- **Robuste** : Gestion d'erreurs complète
- **Extensible** : Facile d'ajouter de nouvelles fonctionnalités
- **Maintenable** : Architecture claire et modulaire

La base de données SQLite est la solution idéale pour votre cas d'usage : gratuite, locale, performante et parfaitement adaptée à un système RAG étudiant.

