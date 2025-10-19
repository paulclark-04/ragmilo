import argparse
import json
import sys
from pathlib import Path

import ollama

from output_formatter import format_response
from rag_core import DEFAULT_EMBEDDING_MODEL, HybridRetriever


LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
DEFAULT_THRESHOLD = 0.35


def main():
    global LANGUAGE_MODEL

    parser = argparse.ArgumentParser(description='ECE Paris RAG (hybride FAISS + BM25)')
    parser.add_argument('--question', '-q', help='Question to ask')
    parser.add_argument('--matiere', help='Filter by matiere')
    parser.add_argument('--enseignant', help='Filter by enseignant')
    parser.add_argument('--semestre', help='Filter by semestre')
    parser.add_argument('--promo', help='Filter by promo')
    parser.add_argument('--top-n', type=int, default=3, help='Top N chunks to retrieve')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD, help='Min vector similarity to answer')
    parser.add_argument('--embed-model', default=DEFAULT_EMBEDDING_MODEL, help='Embedding model id for Ollama')
    parser.add_argument('--llm-model', default=LANGUAGE_MODEL, help='Language model id for Ollama')
    parser.add_argument('--json', action='store_true', help='Output JSON only (no streamed text)')
    parser.add_argument('--alpha', type=float, default=0.65, help='Weight between vector (1.0) and BM25 (0.0) scores')
    parser.add_argument('--vector-k', type=int, default=20, help='Number of FAISS candidates to pull before fusion')
    parser.add_argument('--bm25-k', type=int, default=40, help='Number of BM25 candidates to pull before fusion')
    parser.add_argument('--vector-db', default='vector_db.json', help='Path to vector_db.json')
    parser.add_argument('--faiss-index', default='vector_index.faiss', help='Path to FAISS index file')
    parser.add_argument('--bm25-index', default='bm25_index.pkl', help='Path to BM25 index file')
    parser.add_argument('--meta', default='index_meta.json', help='Path to index metadata file')
    args = parser.parse_args()

    LANGUAGE_MODEL = args.llm_model

    retriever = HybridRetriever(
        vector_db_path=Path(args.vector_db),
        faiss_path=Path(args.faiss_index),
        bm25_path=Path(args.bm25_index),
        meta_path=Path(args.meta),
        embedding_model=args.embed_model,
    )

    input_query = args.question or input('Pose ta question: ')

    metadata_filter = {
        'matiere': args.matiere,
        'enseignant': args.enseignant,
        'semestre': args.semestre,
        'promo': args.promo,
    }
    metadata_filter = {k: v for k, v in metadata_filter.items() if v is not None}

    retrieved_knowledge = retriever.retrieve(
        input_query,
        top_n=args.top_n,
        metadata_filter=metadata_filter or None,
        vector_k=args.vector_k,
        bm25_k=args.bm25_k,
        alpha=args.alpha,
    )

    best_vector = 0.0
    for _, _, meta in retrieved_knowledge:
        best_vector = max(best_vector, meta.get('vector_score', 0.0))

    if best_vector < args.threshold:
        answer = 'Information non trouvée dans les sources disponibles.'
        json_output = format_response(
            answer,
            retrieved_knowledge,
            metadata_filter,
            retrieval_threshold=args.threshold,
        )
        if args.json:
            print(json_output)
        else:
            result = json.loads(json_output)
            print(f"\nQuestion: {input_query}\n")
            print('Réponse:')
            print(answer)
            print('\nSources:')
            if result['sources']:
                for source in result['sources']:
                    label = source.get('doc_label') or source['doc_id']
                    print(
                        f" - {label} p.{source['page']}#{source['chunk_index']}"
                        f" (score {source['score']} | vect {source['vector_score']} | bm25 {source['lexical_score']})"
                    )
            else:
                print(' - Aucune source (score sous le seuil)')
            print(f"Indice de confiance: {result['confidence']}")
        sys.exit(0)

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
        "Si tu ne trouves pas la réponse exacte, réponds: \"Information non trouvée dans les sources disponibles.\"\n\n"
        f"Extraits autorisés:\n{context_block}\n"
    )

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=not args.json,
    )

    answer = ""
    if args.json:
        if isinstance(stream, dict):
            answer = stream.get('message', {}).get('content', '')
        else:
            for chunk in stream:
                content = chunk['message']['content']
                answer += content
    else:
        print(f"\nQuestion: {input_query}\n")
        print('Réponse:')
        for chunk in stream:
            content = chunk['message']['content']
            print(content, end='', flush=True)
            answer += content
        print()

    json_output = format_response(
        answer,
        retrieved_knowledge,
        metadata_filter,
        retrieval_threshold=args.threshold,
    )

    if args.json:
        print(json_output)
    else:
        result = json.loads(json_output)
        print('\nSources:')
        for source in result['sources']:
            label = source.get('doc_label') or source['doc_id']
            print(
                f" - {label} p.{source['page']}#{source['chunk_index']}"
                f" (score {source['score']} | vect {source['vector_score']} | bm25 {source['lexical_score']})"
            )
        if not result['sources']:
            print(' - Aucune source')
        print(f"Indice de confiance: {result['confidence']}")


if __name__ == '__main__':
    main()
