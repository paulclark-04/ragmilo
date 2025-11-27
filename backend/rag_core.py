import hashlib
import json
import pickle
import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import ollama


try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit('Faiss est requis. Installez-le avec `pip install faiss-cpu`.') from exc

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit('rank_bm25 est requis. Installez-le avec `pip install rank-bm25`.') from exc


DEFAULT_EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'

BASE_DIR = Path(__file__).resolve().parent

def tokenize(text: str) -> List[str]:
    pattern = re.compile(r"\b\w+\b", re.UNICODE)
    return pattern.findall(text.lower())


def metadata_matches(meta: Dict, filters: Optional[Dict]) -> bool:
    if not filters:
        return True
    return all(meta.get(k) == v for k, v in filters.items())


def normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return {k: 1.0 for k in scores}
    adjusted = {k: (v - minimum) for k, v in scores.items()}
    scale = max(adjusted.values()) or 1.0
    return {k: val / scale for k, val in adjusted.items()}


def _ensure_identifiers(metadata: Dict, chunk: str) -> Dict:
    meta = metadata.copy()
    doc_id = meta.get('doc_id')
    if not doc_id:
        fallback = meta.get('matiere') or 'doc'
        chunk_hash = hashlib.sha1(chunk.encode('utf-8')).hexdigest()[:8]
        doc_id = f"{fallback}-{chunk_hash}"
        meta['doc_id'] = doc_id

    doc_label = meta.get('doc_label') or doc_id
    meta['doc_label'] = doc_label

    page = meta.get('page', '?')
    meta['page'] = page

    chunk_index = meta.get('chunk_index')
    if isinstance(chunk_index, str):
        try:
            chunk_index = int(chunk_index)
        except ValueError:
            chunk_index = None
    if chunk_index is None:
        chunk_index = 0
    meta['chunk_index'] = chunk_index

    chunk_id = meta.get('chunk_id')
    if not chunk_id:
        meta['chunk_id'] = f"{doc_id}:{page}:{chunk_index}"

    return meta


class HybridRetriever:
    def __init__(
        self,
        vector_db_path: Path = BASE_DIR / 'vector_db.json',
        faiss_path: Path = BASE_DIR / 'vector_index.faiss',
        bm25_path: Path = BASE_DIR / 'bm25_index.pkl',
        meta_path: Path = BASE_DIR / 'index_meta.json',
        embedding_model: Optional[str] = None,
    ) -> None:
        self.vector_db_path = vector_db_path
        self.faiss_path = faiss_path
        self.bm25_path = bm25_path
        self.meta_path = meta_path

        self.documents = self._load_documents()
        self.faiss_index = self._load_faiss_index()
        self.bm25 = self._load_bm25()
        self.meta = self._load_meta()

        self.embedding_model = (
            embedding_model
            or self.meta.get('embedding_model')
            or DEFAULT_EMBEDDING_MODEL
        )

    def _load_documents(self) -> List[Dict]:
        if not self.vector_db_path.exists():
            raise SystemExit(
                f"{self.vector_db_path} introuvable. Lancez d’abord `ingest_pdf.py`."
            )
        with self.vector_db_path.open('r', encoding='utf-8') as fh:
            raw = json.load(fh)
        docs = []
        for item in raw:
            docs.append({
                'text': item['text'],
                'metadata': item.get('metadata', {}),
                'embedding': item.get('embedding'),
            })
        return docs

    def _load_faiss_index(self):
        if not self.faiss_path.exists():
            raise SystemExit(
                f'Index FAISS introuvable ({self.faiss_path}). Relancez `ingest_pdf.py`.'
            )
        index = faiss.read_index(str(self.faiss_path))
        if index.ntotal != len(self.documents):
            print(
                (
                    f"[Avertissement] Index FAISS contient {index.ntotal} vecteurs, "
                    f"mais {len(self.documents)} documents chargés."
                ),
                file=sys.stderr,
            )
        return index

    def _load_bm25(self):
        if not self.bm25_path.exists():
            raise SystemExit(
                f'Index BM25 introuvable ({self.bm25_path}). Relancez `ingest_pdf.py`.'
            )
        with self.bm25_path.open('rb') as fh:
            payload = pickle.load(fh)
        bm25 = payload.get('bm25')
        if bm25 is None:
            raise SystemExit('bm25_index.pkl invalide: objet BM25 absent.')
        return bm25

    def _load_meta(self) -> Dict:
        if not self.meta_path.exists():
            return {}
        with self.meta_path.open('r', encoding='utf-8') as fh:
            return json.load(fh)

    def _prepare_query_embedding(self, query: str) -> np.ndarray:
        response = ollama.embed(model=self.embedding_model, input=query)
        embedding = np.array(response['embeddings'][0], dtype='float32')
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            embedding = np.zeros_like(embedding)
        return embedding

    def retrieve(
        self,
        query: str,
        top_n: int = 3,
        metadata_filter: Optional[Dict] = None,
        vector_k: int = 20,
        bm25_k: int = 40,
        alpha: float = 0.65,
    ) -> List[Tuple[str, float, Dict]]:
        query_vec = self._prepare_query_embedding(query)

        vector_scores_raw: Dict[int, float] = {}
        if self.faiss_index.ntotal:
            k = min(vector_k, self.faiss_index.ntotal)
            distances, indices = self.faiss_index.search(query_vec.reshape(1, -1), k)
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                meta = self.documents[idx]['metadata']
                if metadata_filter and not metadata_matches(meta, metadata_filter):
                    continue
                vector_scores_raw[idx] = float(score)

        lexical_scores_raw: Dict[int, float] = {}
        query_tokens = tokenize(query) or query.lower().split()

        bm25_scores = self.bm25.get_scores(query_tokens)
        if bm25_scores is not None:
            bm25_array = np.array(bm25_scores, dtype='float32')
            k = min(bm25_k, len(bm25_array))
            if k > 0:
                top_indices = np.argpartition(bm25_array, -k)[-k:]
                for idx in top_indices:
                    score = float(bm25_array[idx])
                    if score <= 0:
                        continue
                    meta = self.documents[idx]['metadata']
                    if metadata_filter and not metadata_matches(meta, metadata_filter):
                        continue
                    lexical_scores_raw[idx] = score

        norm_vector = normalize_scores(vector_scores_raw)
        norm_lexical = normalize_scores(lexical_scores_raw)

        candidates = set(norm_vector.keys()) | set(norm_lexical.keys())
        if not candidates and vector_scores_raw:
            candidates = set(vector_scores_raw.keys())

        results: List[Tuple[str, float, Dict]] = []
        for idx in candidates:
            combined = alpha * norm_vector.get(idx, 0.0) + (1 - alpha) * norm_lexical.get(idx, 0.0)
            meta = self.documents[idx]['metadata'].copy()
            meta['vector_score'] = vector_scores_raw.get(idx, 0.0)
            meta['lexical_score'] = lexical_scores_raw.get(idx, 0.0)
            meta['combined_score'] = combined
            meta = _ensure_identifiers(meta, self.documents[idx]['text'])
            results.append((self.documents[idx]['text'], combined, meta))

        results.sort(key=lambda x: x[1], reverse=True)

        if len(results) < top_n:
            extra_indices = [idx for idx in vector_scores_raw if idx not in candidates]
            for idx in extra_indices:
                meta = self.documents[idx]['metadata'].copy()
                meta['vector_score'] = vector_scores_raw.get(idx, 0.0)
                meta['lexical_score'] = lexical_scores_raw.get(idx, 0.0)
                meta['combined_score'] = norm_vector.get(idx, 0.0)
                meta = _ensure_identifiers(meta, self.documents[idx]['text'])
                results.append((self.documents[idx]['text'], norm_vector.get(idx, 0.0), meta))
                if len(results) >= top_n:
                    break

        return results[:top_n]


__all__ = [
    'HybridRetriever',
    'DEFAULT_EMBEDDING_MODEL',
    'tokenize',
]
