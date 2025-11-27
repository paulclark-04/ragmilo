"""
Enhanced PDF ingestion script with database integration
Combines the original ingest_pdf.py functionality with database management
"""

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import ollama
import fitz  # PyMuPDF

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit('Faiss est requis. Installez-le avec `pip install faiss-cpu`.') from exc

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:  # pragma: no cover
    raise SystemExit('rank_bm25 est requis. Installez-le avec `pip install rank_bm25`.') from exc

from backend.database_manager import DatabaseManager, export_to_vector_db


DEFAULT_EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    print(f"[DEBUG] Chunking text of length {len(text)}")
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = ' '.join(words[i:i + size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks


def tokenize(text: str) -> List[str]:
    """Tokenize text for BM25"""
    return TOKEN_PATTERN.findall(text.lower())


def process_pdf_file(
    pdf_path: str,
    metadata: Dict,
    embedding_model: str,
    db_manager: DatabaseManager,
    file_id: Optional[int] = None
) -> int:
    """
    Process a PDF file and store in database.

    Args:
        pdf_path (str): Path to the PDF file.
        metadata (Dict): Classification metadata (matiere, sous_matiere, etc.).
        embedding_model (str): Ollama embedding model.
        db_manager (DatabaseManager): Database manager instance.
        file_id (Optional[int]): Existing file ID if already created.

    Returns:
        Tuple[int, List[List[str]]]: File ID and BM25 corpus.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ✅ Use provided file_id or add a new file entry
    if not file_id:
        file_id = db_manager.add_file(
            file_path=str(pdf_path),
            matiere=metadata.get("matiere"),
            sous_matiere=metadata.get("sous_matiere"),
            enseignant=metadata.get("enseignant"),
            semestre=metadata.get("semestre"),
            promo=metadata.get("promo"),
            doc_id=metadata.get("doc_id"),
            doc_label=metadata.get("doc_label", pdf_path.stem)
        )

    print(f"Added or using existing file: {pdf_path.name} (ID: {file_id})")

    # ✅ Process PDF
    doc = fitz.open(pdf_path)
    chunks_data = []
    bm25_corpus = []

    total_pages = len(doc)
    for page_num in range(total_pages):
        text = doc[page_num].get_text()
        if not text.strip():
            continue  # Skip empty pages

        page_chunks = chunk_text(text)

        for idx, chunk in enumerate(page_chunks):
            # Generate embedding
            embedding = ollama.embed(model=embedding_model, input=chunk)["embeddings"][0]
            embedding_array = np.array(embedding, dtype=np.float32)

            # Build chunk data
            chunk_id = f"{metadata.get('doc_id', pdf_path.stem)}:{page_num + 1}:{idx}"
            chunk_data = {
                "chunk_id": chunk_id,
                "chunk_text": chunk,
                "page_number": page_num + 1,
                "chunk_index": idx,
                "embedding": embedding_array,
                "embedding_model": embedding_model,
                "matiere": metadata.get("matiere"),
                "sous_matiere": metadata.get("sous_matiere"),
                "enseignant": metadata.get("enseignant"),
                "semestre": metadata.get("semestre"),
                "promo": metadata.get("promo")
            }

            chunks_data.append(chunk_data)
            bm25_corpus.append(tokenize(chunk))

        print(f"Processed page {page_num + 1}/{total_pages}", end="\r")

    doc.close()

    print(f"[DEBUG] {len(chunks_data)} chunks ready for DB insert.")
    if len(chunks_data) > 0:
        print(f"Example chunk: {chunks_data[0]['chunk_text'][:200]}")

        # ✅ Store chunks in database
        db_manager.add_rag_chunks(file_id, chunks_data)
        db_manager.mark_file_processed(file_id, len(chunks_data))

        print(f"\nProcessed {len(chunks_data)} chunks from {pdf_path.name}")
        return file_id, bm25_corpus


def build_faiss_index_from_db(db_manager: DatabaseManager, output_path: str):
    """Build FAISS index from database chunks (robust handling of embedding formats)

    NOTE:
      - Certains embeddings peuvent être stockés sous forme de liste de floats,
        de bytes (buffer), ou d'autres formats. Cette version essaye de
        normaliser tous les embeddings en tableaux numpy de dtype float32,
        filtre les embeddings invalides et conserve la dimension la plus
        fréquente si nécessaire.
    """
    chunks = db_manager.get_rag_chunks_by_classification()

    if not chunks:
        print("No chunks found in database")
        return

    embeddings = []
    bad_count = 0
    dims = []

    for idx, chunk in enumerate(chunks):
        raw_emb = chunk.get('embedding', None)

        # Skip missing embeddings
        if raw_emb is None:
            bad_count += 1
            continue

        try:
            # Case 1: embedding already a numpy-compatible sequence (list/tuple)
            if isinstance(raw_emb, (list, tuple)):
                arr = np.array(raw_emb, dtype=np.float32)

            # Case 2: embedding stored as bytes/bytearray (raw buffer)
            elif isinstance(raw_emb, (bytes, bytearray)):
                # Assumer float32 little-endian (typique quand on stocke buffers)
                arr = np.frombuffer(raw_emb, dtype=np.float32)

            # Case 3: embedding serialized as string (e.g., JSON string)
            elif isinstance(raw_emb, str):
                # Essayer de parser la string en list si c'est du JSON
                try:
                    parsed = json.loads(raw_emb)
                    arr = np.array(parsed, dtype=np.float32)
                except Exception:
                    # Dernière chance : essayer conversion numpy directe
                    try:
                        arr = np.array(json.loads(raw_emb), dtype=np.float32)
                    except Exception:
                        continue

            # Fallback : tenter conversion brute-force
            else:
                arr = np.array(raw_emb, dtype=np.float32)

            # Vérifier que l'embedding n'est pas vide et est 1D
            if arr.size == 0 or arr.ndim != 1:
                bad_count += 1
                continue

            embeddings.append(arr)
            dims.append(arr.shape[0])

        except Exception as e:
            # Log et continuer (ne pas interrompre la construction de l'index)
            print(f"Warning: failed to convert embedding for chunk index {idx}: {e}")
            bad_count += 1
            continue

    if not embeddings:
        print("No valid embeddings found")
        return

    # Vérifier la consistance dimensionnelle
    unique_dims = set(dims)
    if len(unique_dims) > 1:
        from collections import Counter
        counter = Counter(dims)
        most_common_dim, count = counter.most_common(1)[0]
        print(f"Dimension inconsistency detected: {dict(counter)}. Keeping embeddings of dim={most_common_dim} ({count}/{len(dims)}).")

        # Filtrer pour ne garder que la dimension la plus fréquente
        filtered_embeddings = [e for e in embeddings if e.shape[0] == most_common_dim]
        if not filtered_embeddings:
            print("After filtering by most common dimension, no embeddings remain.")
            return
        matrix = np.vstack(filtered_embeddings).astype(np.float32)

    else:
        # Toutes les dimensions identiques — empiler directement
        matrix = np.vstack(embeddings).astype(np.float32)

    # Normaliser et construire l'index FAISS
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, output_path)
    print(f"Built FAISS index with {index.ntotal} vectors (skipped {bad_count} invalid embeddings)")



def build_bm25_index_from_db(db_manager: DatabaseManager, output_path: str):
    """Build BM25 index from database chunks"""
    chunks = db_manager.get_rag_chunks_by_classification()
    
    if not chunks:
        print("No chunks found in database")
        return
    
    corpus = []
    for chunk in chunks:
        tokens = tokenize(chunk['chunk_text'])
        corpus.append(tokens)
    
    if not corpus:
        print("No corpus found")
        return
    
    bm25 = BM25Okapi(corpus)
    with open(output_path, 'wb') as fh:
        pickle.dump({'bm25': bm25}, fh)
    print(f"Built BM25 index with {len(corpus)} documents")


def save_meta(path: str, chunk_count: int, embedding_model: str):
    """Save metadata about the index"""
    meta = {
        'embedding_model': embedding_model,
        'chunk_count': chunk_count,
        'database_integrated': True
    }
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"Saved metadata to {path}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced PDF ingestion with database integration')
    parser.add_argument('--pdf', action='append', required=True, help='Path to a PDF (can be repeated)')
    parser.add_argument('--matiere', required=True, help='Subject (matière)')
    parser.add_argument('--sous_matiere', required=True, help='Sub-subject (sous-matière)')
    parser.add_argument('--enseignant', required=True, help='Instructor')
    parser.add_argument('--promo', required=True, help='Promo/Year, e.g., 2025')
    parser.add_argument('--semestre', required=True, help='Semester, e.g., S1')
    parser.add_argument('--file_id', type=int, help='Existing file ID in database (optional)')
    parser.add_argument('--embed-model', default=DEFAULT_EMBEDDING_MODEL, help='Ollama embedding model id')
    parser.add_argument('--output', default='vector_db.json', help='Output JSON DB path')
    parser.add_argument('--faiss-index', default='vector_index.faiss', help='FAISS index output path')
    parser.add_argument('--bm25-index', default='bm25_index.pkl', help='BM25 index output path')
    parser.add_argument('--meta-output', default='index_meta.json', help='Index metadata output path')
    parser.add_argument('--db-path', default='rag_database.db', help='SQLite database path')
    parser.add_argument('--export-only', action='store_true', help='Only export from database, don\'t process new files')
    parser.add_argument('--import-existing', action='store_true', help='Import existing vector_db.json into database')
    
    args = parser.parse_args()
    
    with DatabaseManager(args.db_path) as db_manager:
        
        if args.import_existing:
            # Import existing vector_db.json into database
            if Path(args.output).exists():
                print("Importing existing vector_db.json into database...")
                from backend.database_manager import import_from_vector_db
                import_from_vector_db(db_manager, args.output)
            else:
                print(f"File not found: {args.output}")
                return
        
        if not args.export_only:
            # Process new PDFs
            base_metadata = {
                'matiere': args.matiere,
                'sous_matiere': args.sous_matiere,
                'enseignant': args.enseignant,
                'semestre': args.semestre,
                'promo': args.promo,
            }
            if args.file_id:
                print(f"Using existing file ID: {args.file_id}")
                
            all_bm25_corpus = []






            for pdf_path in args.pdf:
                # Generate doc_id for this PDF
                pdf_name = Path(pdf_path).stem
                doc_id = f"{args.matiere.lower().replace(' ', '-')}-{pdf_name.lower()}"
                
                pdf_metadata = base_metadata.copy()
                pdf_metadata['doc_id'] = doc_id
                pdf_metadata['doc_label'] = pdf_name
                
                try:
                    file_id, bm25_corpus = process_pdf_file(
                        pdf_path, pdf_metadata, args.embed_model, db_manager
                    )
                    all_bm25_corpus.extend(bm25_corpus)
                    print(f"Successfully processed: {pdf_path}")
                except Exception as e:
                    print(f"Error processing {pdf_path}: {e}")
        
        # Export to vector_db.json format for compatibility
        print("Exporting to vector_db.json...")
        export_to_vector_db(db_manager, "vector_db.json")
        
        # Build indexes
        print("Building FAISS index...")
        build_faiss_index_from_db(db_manager, args.faiss_index)
        
        print("Building BM25 index...")
        build_bm25_index_from_db(db_manager, args.bm25_index)
        
        # Save metadata
        chunks = db_manager.get_rag_chunks_by_classification()
        save_meta(args.meta_output, len(chunks), args.embed_model)
        
        # Show summary
        summary = db_manager.get_file_summary()
        print(f"\nDatabase summary:")
        print(f"  Total files: {len(summary)}")
        print(f"  Total chunks: {sum(f['actual_chunks'] for f in summary)}")
        
        classifications = db_manager.get_unique_classifications()
        print(f"\nAvailable classifications:")
        for field, values in classifications.items():
            print(f"  {field}: {values}")


if __name__ == '__main__':
    main()

