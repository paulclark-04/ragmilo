import json
from pathlib import Path
from typing import Dict, Optional

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from output_formatter import format_response
from rag_core import DEFAULT_EMBEDDING_MODEL, HybridRetriever


frontend_dir = Path('../frontend/front_voice')

app = FastAPI(title='ECE RAG API', version='1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class QueryRequest(BaseModel):
    question: str
    matiere: Optional[str] = None
    enseignant: Optional[str] = None
    semestre: Optional[str] = None
    promo: Optional[str] = None
    top_n: int = 3
    threshold: float = 0.35
    alpha: float = 0.65
    vector_k: int = 20
    bm25_k: int = 40
    embed_model: Optional[str] = None
    llm_model: str = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'


class Query(BaseModel):
    question: str


retriever: Optional[HybridRetriever] = None


def ensure_retriever(embed_model: Optional[str] = None) -> HybridRetriever:
    global retriever
    if retriever is None or (embed_model and retriever.embedding_model != embed_model):
        retriever = HybridRetriever(embedding_model=embed_model or DEFAULT_EMBEDDING_MODEL)
    return retriever



def build_prompt(retrieved_knowledge, threshold):
    context_lines = []
    for chunk, score, meta in retrieved_knowledge:
        cid = meta.get('chunk_id') or f"{meta.get('doc_id','?')}:{meta.get('page','?')}:{meta.get('chunk_index','?')}"
        context_lines.append(f"[{cid}] (score {score:.2f}) {chunk}")
    context_block = '\n'.join(context_lines)

    instruction_prompt = (
        "Tu es un assistant pédagogique ECE Paris. Réponds exclusivement en français clair et concis.\n"
        "Utilise UNIQUEMENT les informations présentes dans les extraits ci-dessous.\n"
        "Chaque affirmation doit être suivie de la citation de la forme [docid:page:index].\n"
        "Ne crée ni exemples, ni équations, ni explications absents des extraits.\n"
        f"Si tu ne trouves pas la réponse exacte, réponds: \"Information non trouvée dans les sources disponibles.\" (seuil {threshold}).\n\n"
        f"Extraits autorisés:\n{context_block}\n"
    )
    return instruction_prompt



@app.get('/api/metadata')
def get_metadata():
    retr = ensure_retriever()
    keys = ['matiere', 'enseignant', 'semestre', 'promo']
    unique: Dict[str, set] = {k: set() for k in keys}
    records = []
    seen = set()

    for doc in retr.documents:
        meta = doc['metadata']
        record = {k: meta.get(k) for k in keys}
        key = tuple(record.get(k) for k in keys)
        if key not in seen:
            seen.add(key)
            records.append(record)
        for k, val in record.items():
            if val:
                unique[k].add(val)

    return {
        'unique': {k: sorted(v) for k, v in unique.items()},
        'records': records,
    }



@app.post('/api/ask')
def ask_question(payload: QueryRequest):
    retr = ensure_retriever(payload.embed_model)

    metadata_filter = {
        'matiere': payload.matiere,
        'enseignant': payload.enseignant,
        'semestre': payload.semestre,
        'promo': payload.promo,
    }
    metadata_filter = {k: v for k, v in metadata_filter.items() if v}

    retrieved = retr.retrieve(
        payload.question,
        top_n=payload.top_n,
        metadata_filter=metadata_filter or None,
        alpha=payload.alpha,
        vector_k=payload.vector_k,
        bm25_k=payload.bm25_k,
    )

    best_vector = 0.0
    for _, _, meta in retrieved:
        best_vector = max(best_vector, meta.get('vector_score', 0.0))

    if best_vector < payload.threshold:
        answer = "Information non trouvée dans les sources disponibles."
        response = json.loads(
            format_response(answer, retrieved, metadata_filter, retrieval_threshold=payload.threshold)
        )
        return response

    prompt = build_prompt(retrieved, payload.threshold)

    try:
        chat = ollama.chat(
            model=payload.llm_model,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': payload.question},
            ],
            stream=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    answer = chat.get('message', {}).get('content', '')

    response = json.loads(
        format_response(answer, retrieved, metadata_filter, retrieval_threshold=payload.threshold)
    )
    return response



@app.post("/rag/query")
async def rag_query(q: Query):
    """Ancien endpoint utilisé par Milo, désormais redirigé vers /api/ask."""
    req = QueryRequest(question=q.question)
    return ask_question(req)



if frontend_dir.exists():
    app.mount('/', StaticFiles(directory=frontend_dir, html=True), name='frontend')
