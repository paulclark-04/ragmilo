import json
import webbrowser
import threading
from pathlib import Path
from typing import Dict, Optional

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.output_formatter import format_response
from backend.rag_core import DEFAULT_EMBEDDING_MODEL, HybridRetriever


BASE_DIR = Path(__file__).resolve().parent.parent 
FRONTEND_DIR = BASE_DIR / "frontend"

FRONT_VOICE_DIR = FRONTEND_DIR / "front_voice"
FRONT_TEXT_DIR = FRONTEND_DIR / "front_text"



app = FastAPI(title='ECE RAG API', version='1.0')
app.mount("/front_text", StaticFiles(directory="frontend/front_text"), name="front_text")
app.mount("/front_voice", StaticFiles(directory="frontend/front_voice"), name="front_voice")

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
    llm_model: str = 'mistral:7b'


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
        "Tu es un assistant pédagogique ECE Paris. RÈGLES STRICTES (INTERDICTION ABSOLUE DE LES ENFREINDRE):\\n\\n"
        
        "✅ CE QUE TU DOIS FAIRE:\\n"
        "1. Réponds UNIQUEMENT en français clair et concis\\n"
        "2. Utilise EXCLUSIVEMENT les informations textuellement présentes dans les extraits ci-dessous\\n"
        "4. Si l'information n'est PAS EXPLICITEMENT dans les extraits, dis EXACTEMENT : "
        "\\\"Je ne trouve pas cette information dans les documents disponibles.\\\"\\n\\n"
        
        "❌ INTERDICTIONS ABSOLUES:\\n"
        "1. NE génère JAMAIS de contenu qui n'est pas littéralement dans les extraits\\n"
        "2. NE crée AUCUN exemple, équation, code, ou explication de ton propre chef\\n"
        "3. NE fais AUCUNE déduction ou inférence au-delà du texte exact\\n"
        "4. NE combine PAS d'informations de sources différentes pour créer des faits\\n"
        "5. NE réponds JAMAIS si tu n'es pas sûr à 100% que c'est dans les extraits\\n"
        
        "⚠️ EN CAS DE DOUTE : Dis que tu ne sais pas. C'est PRÉFÉRABLE à une réponse incertaine.\\n\\n"
        
        f"EXTRAITS AUTORISÉS (seuil de confiance: {threshold}):\\n"
        f"{context_block}\\n\\n"
        
        "RAPPEL : Si la réponse n'est pas EXPLICITEMENT et CLAIREMENT dans les extraits ci-dessus, "
        "réponds : \\\"Je ne trouve pas cette information dans les documents disponibles.\\\""
    )
    return instruction_prompt


@app.get('/api/metadata')
def get_metadata():
    retr = ensure_retriever()
    keys = ['matiere', 'sous_matiere', 'enseignant', 'semestre', 'promo']
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


@app.post('/api/reload')
def reload_data():
    """Force reload of RAG data"""
    global retriever
    retriever = None
    print("[INFO] RAG Data reload requested.")
    return {"message": "Data reload initiated"}


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

    best_vector = max((meta.get('vector_score', 0.0) for _, _, meta in retrieved), default=0.0)

    if best_vector < payload.threshold:
        answer = "Information non trouvée dans les sources disponibles."
        response = json.loads(format_response(answer, retrieved, metadata_filter, retrieval_threshold=payload.threshold))
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

    response = json.loads(format_response(answer, retrieved, metadata_filter, retrieval_threshold=payload.threshold))
    return response


@app.post("/rag/query")
async def rag_query(q: Query):
    req = QueryRequest(question=q.question)
    return ask_question(req)




if FRONT_TEXT_DIR.exists():
    app.mount("/front_text", StaticFiles(directory=str(FRONT_TEXT_DIR), html=True), name="front_text")

if FRONT_VOICE_DIR.exists():
    app.mount("/front_voice", StaticFiles(directory=str(FRONT_VOICE_DIR), html=True), name="front_voice")



def open_frontend():
    webbrowser.open("http://127.0.0.1:8000/front_voice/index_voice.html")

threading.Timer(1.0, open_frontend).start()
