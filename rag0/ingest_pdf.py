import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

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
    raise SystemExit('rank_bm25 est requis. Installez-le avec `pip install rank-bm25`.') from exc

from improved_chunking import improved_chunk_text

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
VECTOR_DB = []
EMBEDDINGS = []
BM25_CORPUS = []
EXISTING_CHUNK_IDS = set()
DOC_IDS = set()

TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def chunk_text(text, size=500, overlap=50):
    """
    Wrapper pour improved_chunk_text avec méthode 'paragraphs'.
    L'ancienne fonction est conservée pour compatibilité mais utilise maintenant
    le chunking amélioré par paragraphes.

        method: Méthode de chunking
            - 'paragraphs': Découpe par paragraphes (RECOMMANDÉ)
            - 'sections': Découpe par sections avec détection de titres
            - 'context': Découpe avec contexte de section (MEILLEUR)
            - 'smart': Overlap intelligent avec respect des phrases
    """
    return improved_chunk_text(text, size=size, method='context')


def tokenize(text: str):
    return TOKEN_PATTERN.findall(text.lower())


def _generate_doc_id(pdf_path: str) -> str:
    p = Path(pdf_path)
    base = p.stem.lower()
    base = re.sub(r'[^a-z0-9]+', '-', base).strip('-') or 'doc'

    candidate = base
    suffix = 1
    while candidate in DOC_IDS:
        suffix += 1
        candidate = f"{base}-{suffix}"

    DOC_IDS.add(candidate)
    return candidate


def parse_pdf(pdf_path, metadata):
    doc_label = metadata.get('doc_label') or Path(pdf_path).stem
    doc_id = metadata.get('doc_id') or _generate_doc_id(pdf_path)
    metadata = metadata.copy()
    metadata['doc_id'] = doc_id
    metadata['doc_label'] = doc_label

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    start_count = len(VECTOR_DB)
    for page_num in range(total_pages):
        text = doc[page_num].get_text()
        # Utilise le chunking amélioré par paragraphes
        page_chunks = improved_chunk_text(text, size=500, method='paragraphs')

        for idx, chunk in enumerate(page_chunks):
            embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
            meta = metadata.copy()
            meta['page'] = page_num + 1
            meta['chunk_index'] = idx
            chunk_id = f"{doc_id}:{page_num + 1}:{idx}"
            if chunk_id in EXISTING_CHUNK_IDS:
                continue
            meta['chunk_id'] = chunk_id
            VECTOR_DB.append((chunk, embedding, meta))
            EMBEDDINGS.append(embedding)
            BM25_CORPUS.append(tokenize(chunk))
            EXISTING_CHUNK_IDS.add(chunk_id)
        print(f"Processed page {page_num + 1}/{total_pages}", end='\r')

    doc.close()
    added = len(VECTOR_DB) - start_count
    print(f"\nAdded {added} chunks from {metadata.get('matiere', 'unknown')} ({pdf_path})")


def save_db(filename='vector_db.json'):
    data = [{'text': c, 'embedding': e, 'metadata': m} for c, e, m in VECTOR_DB]
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Saved {len(VECTOR_DB)} chunks to {filename}")


def build_faiss_index(output_path: str):
    if not EMBEDDINGS:
        print('No embeddings collected; skipping FAISS index build.')
        return

    matrix = np.array(EMBEDDINGS, dtype='float32')
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, output_path)
    print(f"Saved FAISS index with {index.ntotal} vectors to {output_path}")


def build_bm25_index(output_path: str):
    if not BM25_CORPUS:
        print('No corpus tokens; skipping BM25 index build.')
        return

    bm25 = BM25Okapi(BM25_CORPUS)
    with open(output_path, 'wb') as fh:
        pickle.dump({'bm25': bm25}, fh)
    print(f"Saved BM25 index to {output_path}")


def save_meta(path: str, chunk_count: int):
    meta = {
        'embedding_model': EMBEDDING_MODEL,
        'chunk_count': chunk_count,
    }
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"Saved index metadata to {path}")


def load_existing(output_path: Path, append: bool):
    if not append or not output_path.exists():
        return

    with output_path.open('r', encoding='utf-8') as fh:
        data = json.load(fh)

    for item in data:
        chunk = item['text']
        embedding = item['embedding']
        meta = item.get('metadata', {})
        VECTOR_DB.append((chunk, embedding, meta))
        EMBEDDINGS.append(embedding)
        BM25_CORPUS.append(tokenize(chunk))
        chunk_id = meta.get('chunk_id')
        if chunk_id:
            EXISTING_CHUNK_IDS.add(chunk_id)
        doc_id = meta.get('doc_id')
        if doc_id:
            DOC_IDS.add(doc_id)


def main():
    global EMBEDDING_MODEL

    parser = argparse.ArgumentParser(description='Ingest PDFs and build RAG indexes')
    parser.add_argument('--pdf', action='append', required=True, help='Path to a PDF (can be repeated)')
    parser.add_argument('--matiere', required=True, help='Subject (matière)')
    parser.add_argument('--sous-matiere', dest='sous_matiere', help='Sous-matière (defaults to matiere if omitted)')
    parser.add_argument('--enseignant', required=True, help='Instructor')
    parser.add_argument('--promo', required=True, help='Promo/Year, e.g., 2025')
    parser.add_argument('--semestre', required=True, help='Semester, e.g., S1')
    parser.add_argument('--embed-model', default=EMBEDDING_MODEL, help='Ollama embedding model id')
    parser.add_argument('--output', default='vector_db.json', help='Output JSON DB path')
    parser.add_argument('--faiss-index', default='vector_index.faiss', help='FAISS index output path')
    parser.add_argument('--bm25-index', default='bm25_index.pkl', help='BM25 index output path')
    parser.add_argument('--meta-output', default='index_meta.json', help='Index metadata output path')
    parser.add_argument('--append', action='store_true', help='Append to existing database if present')
    args = parser.parse_args()
    EMBEDDING_MODEL = args.embed_model

    VECTOR_DB.clear()
    EMBEDDINGS.clear()
    BM25_CORPUS.clear()
    EXISTING_CHUNK_IDS.clear()
    DOC_IDS.clear()

    output_path = Path(args.output)
    load_existing(output_path, args.append)

    base_meta = {
        'matiere': args.matiere,
        'sous_matiere': args.sous_matiere or args.matiere,
        'enseignant': args.enseignant,
        'promo': args.promo,
        'semestre': args.semestre,
    }

    for pdf in args.pdf:
        parse_pdf(pdf, base_meta)

    save_db(args.output)
    build_faiss_index(args.faiss_index)
    build_bm25_index(args.bm25_index)
    save_meta(args.meta_output, len(VECTOR_DB))


if __name__ == '__main__':
    main()
