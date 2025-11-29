"""
Database Manager for ECE Paris RAG System
Handles file storage, classification, and integration with RAG pipeline
"""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

class DatabaseManager:
    def __init__(self, db_path: str = None):
        if not db_path:
            db_path = Path(__file__).parent.parent / "data" / "rag_database.db"
        self.db_path = Path(db_path)
        self.connection = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection"""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)

        self.connection.row_factory = sqlite3.Row  # Enable column access by name
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        schema_path = Path(__file__).parent / "database_schema.sql"
        if schema_path.exists():
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = f.read()
            try:
                self.connection.executescript(schema)
                self.connection.commit()
            except sqlite3.OperationalError as e:
                if "already exists" in str(e):
                    # Tables already exist, that's fine
                    pass
                else:
                    raise
    
    def add_file(
        self,
        file_path: str,
        matiere: str,
        sous_matiere: str,
        enseignant: str,
        semestre: str,
        promo: str,
        doc_id: Optional[str] = None,
        doc_label: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> int:
        """
        Add a new file to the database with classification metadata
        
        Returns:
            int: File ID in the database
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Generate file hash for duplicate detection
        file_hash = self._calculate_file_hash(file_path)
        
        # Check for duplicates — reuse existing file instead of raising
        existing = self.get_file_by_hash(file_hash)
        if existing:
            print(f"[ℹ️] Reusing existing file entry (ID={existing['id']}) for {file_path.name}")
            return existing["id"]
        
        # Generate doc_id if not provided
        if not doc_id:
            doc_id = self._generate_doc_id(file_path.stem, matiere)
        
        if not doc_label:
            doc_label = file_path.stem

        sous_matiere = sous_matiere or matiere #Si pas de sous_matiere, init à matiere
        status = "En attente"  # Par défaut avant traitement

        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO files (
                filename, file_path, file_size, file_type,
                matiere, sous_matiere, enseignant, semestre, promo,
                doc_id, doc_label, description, tags, file_hash, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file_path.name,
            str(file_path.absolute()),
            file_path.stat().st_size,
            file_path.suffix.lower(),
            matiere, sous_matiere, enseignant, semestre, promo,
            doc_id, doc_label, description,
            json.dumps(tags) if tags else None,
            file_hash,
            status
        ))
        
        file_id = cursor.lastrowid
        self.connection.commit()
        return file_id
    
    def get_file_by_hash(self, file_hash: str) -> Optional[Dict]:
        """Get file by its hash"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM files WHERE file_hash = ?", (file_hash,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_file_by_id(self, file_id: int) -> Optional[Dict]:
        """Get file by ID"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM files WHERE id = ?", (file_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_files_by_classification(
        self,
        matiere: Optional[str] = None,
        sous_matiere: Optional[str] = None,
        enseignant: Optional[str] = None,
        semestre: Optional[str] = None,
        promo: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict]:
        """Get files filtered by classification criteria"""
        conditions = []
        params = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        if matiere:
            conditions.append("matiere = ?")
            params.append(matiere)
        if sous_matiere:
            conditions.append("sous_matiere = ?")
            params.append(sous_matiere)
        if enseignant:
            conditions.append("enseignant = ?")
            params.append(enseignant)
        if semestre:
            conditions.append("semestre = ?")
            params.append(semestre)
        if promo:
            conditions.append("promo = ?")
            params.append(promo)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        cursor = self.connection.cursor()
        cursor.execute(f"""
            SELECT * FROM files 
            WHERE {where_clause}
            ORDER BY upload_date DESC
        """, params)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_unique_classifications(self) -> Dict[str, List[str]]:
        """Get all unique values for each classification field"""
        cursor = self.connection.cursor()
        
        classifications = {}
        for field in ['matiere', 'sous_matiere', 'enseignant', 'semestre', 'promo']:
            cursor.execute(f"SELECT DISTINCT {field} FROM files WHERE {field} IS NOT NULL ORDER BY {field}")
            classifications[field] = [row[0] for row in cursor.fetchall()]
        
        return classifications
    
    def mark_file_processed(self, file_id: int, chunk_count: int):
        """Mark file as processed and update chunk count"""
        cursor = self.connection.cursor()
        cursor.execute("""
            UPDATE files 
            SET is_processed = TRUE, 
                processing_date = CURRENT_TIMESTAMP,
                chunk_count = ?,
                updated_at = CURRENT_TIMESTAMP,
                status = 'Traite'
            WHERE id = ?
        """, (chunk_count, file_id))
        self.connection.commit()
    
    def add_rag_chunks(self, file_id: int, chunks: List[Dict]):
        """
        Add RAG chunks for a file
        
        Args:
            file_id: Database file ID
            chunks: List of chunk dictionaries with keys:
                - chunk_id, chunk_text, page_number, chunk_index
                - embedding (numpy array), matiere, sous_matiere, enseignant, semestre, promo
        """
        cursor = self.connection.cursor()
        
        for chunk in chunks:
            # Convert numpy embedding to binary
            embedding_blob = chunk['embedding'].tobytes() if isinstance(chunk['embedding'], np.ndarray) else chunk['embedding']
            
            cursor.execute("""
                INSERT INTO rag_chunks (
                    file_id, chunk_id, chunk_text, page_number, chunk_index,
                    embedding, embedding_model, matiere, sous_matiere, enseignant, semestre, promo
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_id,
                chunk['chunk_id'],
                chunk['chunk_text'],
                chunk.get('page_number'),
                chunk.get('chunk_index'),
                embedding_blob,
                chunk.get('embedding_model'),
                chunk['matiere'],
                chunk['sous_matiere'],
                chunk['enseignant'],
                chunk['semestre'],
                chunk['promo']
            ))
        print(f"[DEBUG] Inserting {len(chunks)} chunks for file ID {file_id}") #?? A enlever comme tous les print
        self.connection.commit()
    
    def get_rag_chunks_by_classification(
        self,
        matiere: Optional[str] = None,
        sous_matiere: Optional[str] = None,
        enseignant: Optional[str] = None,
        semestre: Optional[str] = None,
        promo: Optional[str] = None,limit: Optional[int] = None) -> List[Dict]:
        """Get RAG chunks filtered by classification"""
        conditions = []
        params = []
        if matiere:
            conditions.append("c.matiere = ?")
            params.append(matiere)
        if sous_matiere:
            conditions.append("c.sous_matiere = ?")
            params.append(sous_matiere)
        if enseignant:
            conditions.append("c.enseignant = ?")
            params.append(enseignant)
        if semestre:
            conditions.append("c.semestre = ?")
            params.append(semestre)
        if promo:
            conditions.append("c.promo = ?")
            params.append(promo)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        params = [p for p in params if p not in (None, "", "null")]

        cursor = self.connection.cursor()
        query = f"""
            SELECT c.*, f.filename, f.doc_id, f.doc_label
            FROM rag_chunks c
            JOIN files f ON c.file_id = f.id
            WHERE {where_clause}
            ORDER BY c.created_at DESC
            {limit_clause}
        """

        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_file_summary(self) -> List[Dict]:
        """Get summary of all files with processing status"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM file_summary ORDER BY upload_date DESC")
        return [dict(row) for row in cursor.fetchall()]
    
    def delete_file(self, file_id: int) -> bool:
        """Delete a file and all its chunks"""
        cursor = self.connection.cursor()
        
        # Get file info before deletion
        file_info = self.get_file_by_id(file_id)
        if not file_info:
            return False
        
        # Delete chunks first (foreign key constraint)
        cursor.execute("DELETE FROM rag_chunks WHERE file_id = ?", (file_id,))
        
        # Delete file
        cursor.execute("DELETE FROM files WHERE id = ?", (file_id,))
        
        self.connection.commit()
        return True
    
    def update_file_metadata(
        self,
        file_id: int,
        matiere: Optional[str] = None,
        sous_matiere: Optional[str] = None,
        enseignant: Optional[str] = None,
        semestre: Optional[str] = None,
        promo: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Update file metadata"""
        updates = []
        params = []
        
        if matiere is not None:
            updates.append("matiere = ?")
            params.append(matiere)
        if enseignant is not None:
            updates.append("enseignant = ?")
            params.append(enseignant)
        if semestre is not None:
            updates.append("semestre = ?")
            params.append(semestre)
        if promo is not None:
            updates.append("promo = ?")
            params.append(promo)
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
        
        if not updates:
            return False
        
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(file_id)
        
        cursor = self.connection.cursor()
        cursor.execute(f"""
            UPDATE files 
            SET {', '.join(updates)}
            WHERE id = ?
        """, params)
        
        self.connection.commit()
        return cursor.rowcount > 0
    

    def update_file_status(self, file_id: int, status: str):
        """Update file status manually"""
        cursor = self.connection.cursor()
        cursor.execute("""
            UPDATE files
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (status, file_id))
        self.connection.commit()



    def search_files(
        self,
        query: Optional[str] = None,
        matiere: Optional[str] = None,
        sous_matiere: Optional[str] = None,
        enseignant: Optional[str] = None,
        semestre: Optional[str] = None,
        promo: Optional[str] = None
    ) -> List[Dict]:
        """Search files with optional text search and filters"""
        conditions = []
        params = []
        
        if query:
            conditions.append("(filename LIKE ? OR doc_label LIKE ? OR description LIKE ?)")
            query_param = f"%{query}%"
            params.extend([query_param, query_param, query_param])
        
        if matiere:
            conditions.append("matiere = ?")
            params.append(matiere)
        if sous_matiere:
            conditions.append("sous_matiere = ?")
            params.append(sous_matiere)
        if enseignant:
            conditions.append("enseignant = ?")
            params.append(enseignant)
        if semestre:
            conditions.append("semestre = ?")
            params.append(semestre)
        if promo:
            conditions.append("promo = ?")
            params.append(promo)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        cursor = self.connection.cursor()
        cursor.execute(f"""
            SELECT * FROM files 
            WHERE {where_clause}
            ORDER BY upload_date DESC
        """, params)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _generate_doc_id(self, base_name: str, matiere: str) -> str:
        """Generate unique document ID"""
        import re
        
        # Clean base name
        clean_base = re.sub(r'[^a-z0-9]+', '-', base_name.lower()).strip('-')
        clean_matiere = re.sub(r'[^a-z0-9]+', '-', matiere.lower()).strip('-')
        
        # Check for uniqueness
        cursor = self.connection.cursor()
        base_doc_id = f"{clean_matiere}-{clean_base}"
        doc_id = base_doc_id
        counter = 1
        
        while True:
            cursor.execute("SELECT id FROM files WHERE doc_id = ?", (doc_id,))
            if not cursor.fetchone():
                break
            doc_id = f"{base_doc_id}-{counter}"
            counter += 1
        
        return doc_id
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Utility functions for integration with existing RAG system
def export_to_vector_db(db_manager: DatabaseManager, output_path: str = "vector_db.json"):
    """Export database chunks to the existing vector_db.json format"""
    chunks = db_manager.get_rag_chunks_by_classification()
    
    vector_db_data = []
    for chunk in chunks:
        # Convert binary embedding back to numpy array
        embedding = np.frombuffer(chunk['embedding'], dtype=np.float32)

        # Récupération sécurisée avec valeurs par défaut
        metadata = {
            'doc_id': chunk.get('doc_id', ''),
            'doc_label': chunk.get('doc_label', ''),
            'chunk_id': chunk.get('chunk_id', ''),
            'page': chunk.get('page_number', 0),
            'chunk_index': chunk.get('chunk_index', 0),
            'matiere': chunk.get('matiere', ''),
            'sous_matiere': chunk.get('sous_matiere', ''),  # ✅ plus de KeyError
            'enseignant': chunk.get('enseignant', ''),
            'semestre': chunk.get('semestre', ''),
            'promo': chunk.get('promo', ''),
            'filename': chunk.get('filename', '')
        }

        vector_db_data.append({
            'text': chunk.get('chunk_text', ''),
            'embedding': embedding.tolist(),
            'metadata': metadata
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vector_db_data, f, ensure_ascii=False, indent=2)
    
    print(f"Exported {len(vector_db_data)} chunks to {output_path}")


def import_from_vector_db(db_manager: DatabaseManager, vector_db_path: str):
    """Import existing vector_db.json into the database"""
    try:
        with open(vector_db_path, 'r', encoding='utf-8') as f:
            vector_data = json.load(f)
    except UnicodeDecodeError:
        # Try with different encoding
        with open(vector_db_path, 'r', encoding='utf-8-sig') as f:
            vector_data = json.load(f)
    
    # Group chunks by file (assuming doc_id represents files)
    files_data = {}
    for item in vector_data:
        doc_id = item['metadata'].get('doc_id', 'unknown')
        if doc_id not in files_data:
            files_data[doc_id] = {
                'chunks': [],
                'metadata': item['metadata']
            }
        files_data[doc_id]['chunks'].append(item)
    
    for doc_id, file_data in files_data.items():
        metadata = file_data['metadata']
        
        # Add file to database
        try:
            # Create a dummy file path if the original file doesn't exist
            original_filename = metadata.get('filename', f"{doc_id}.pdf")
            if not Path(original_filename).exists():
                # Create a placeholder file for database purposes
                placeholder_path = f"imported_{doc_id}.pdf"
                with open(placeholder_path, 'w') as f:
                    f.write(f"Imported from vector_db.json - Original: {original_filename}")
                file_path = placeholder_path
            else:
                file_path = original_filename
                
            file_id = db_manager.add_file(
                file_path=file_path,
                matiere=metadata.get('matiere', 'Unknown'),
                sous_matiere=metadata.get('sous_matiere', 'Unknown'),
                enseignant=metadata.get('enseignant', 'Unknown'),
                semestre=metadata.get('semestre', 'Unknown'),
                promo=metadata.get('promo', 'Unknown'),
                doc_id=doc_id,
                doc_label=metadata.get('doc_label', doc_id)
            )
            
            # Add chunks
            chunks = []
            for item in file_data['chunks']:
                chunks.append({
                    'chunk_id': item['metadata']['chunk_id'],
                    'chunk_text': item['text'],
                    'page_number': item['metadata'].get('page'),
                    'chunk_index': item['metadata'].get('chunk_index'),
                    'embedding': np.array(item['embedding']),
                    'embedding_model': 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf',
                    'matiere': metadata['matiere'],
                    'sous_matiere': metadata['sous_matiere'],
                    'enseignant': metadata['enseignant'],
                    'semestre': metadata['semestre'],
                    'promo': metadata['promo']
                })
            
            db_manager.add_rag_chunks(file_id, chunks)
            db_manager.mark_file_processed(file_id, len(chunks))
            
            print(f"Imported {doc_id}: {len(chunks)} chunks")
            
        except Exception as e:
            print(f"Error importing {doc_id}: {e}")


if __name__ == "__main__":
    # Example usage
    with DatabaseManager() as db:
        # Get all unique classifications
        classifications = db.get_unique_classifications()
        print("Available classifications:")
        for field, values in classifications.items():
            print(f"  {field}: {values}")
        
        # Get file summary
        summary = db.get_file_summary()
        print(f"\nFile summary: {len(summary)} files")
        for file_info in summary:
            print(f"  {file_info['filename']} - {file_info['matiere']} - {file_info['sous_matiere']} - {file_info['enseignant']}")
