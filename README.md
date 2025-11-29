# RAGMilo - Assistant RAG ECE Paris

Système de Retrieval-Augmented Generation pour ECE Paris utilisant une architecture hybride (FAISS + BM25).

📚 **Documentation complète** : Voir [`docs/`](docs/)

## 🚀 Démarrage Rapide

### Serveur Principal RAG (Port 8000)
```bash
uvicorn backend.server:app --reload
```
Interface : http://127.0.0.1:8000/front_voice/index_voice.html

### Gestionnaire de Fichiers (Port 8001)
```bash
python3 -m backend.file_manager_rag
```
Interface : http://127.0.0.1:8001

## 📂 Structure du Projet

```
ragmilo/
├── docs/              # Documentation
├── backend/           # Code Python
├── data/              # Bases de données et indexes
├── frontend/          # Interface utilisateur
├── audio/             # Fichiers audio
└── synthet iser/      # Synthèse vocale
```

## 📖 Documentation

- [Guide Backend](docs/BACKEND.md) - Architecture et détails techniques
- [Guide Base de Données](docs/DATABASE.md) - Schéma et gestion des données

## 🛠️ Technologies

- **Backend** : FastAPI, Python 3.11+
- **RAG** : FAISS, BM25, Ollama
- **Base de données** : SQLite
- **Frontend** : HTML/CSS/JavaScript
