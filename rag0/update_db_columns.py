import sqlite3

# Chemin vers votre base de données
db_path = "rag_database.db"

# Connexion à la base
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Vérifier si la colonne existe déjà
cursor.execute("PRAGMA table_info(rag_chunks)")
columns = [info[1] for info in cursor.fetchall()]

if "sous_matiere" not in columns:
    # Ajouter la colonne sous_matiere
    cursor.execute("ALTER TABLE rag_chunks ADD COLUMN sous_matiere TEXT")
    print("Colonne 'sous_matiere' ajoutée à rag_chunks.")
else:
    print("La colonne 'sous_matiere' existe déjà.")

# Sauvegarder et fermer
conn.commit()
conn.close()
