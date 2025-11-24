import json


def format_response(answer, retrieved_chunks, metadata_filter_used, retrieval_threshold=None):
    """
    retrieved_chunks: liste de (chunk, similarity, metadata)
    """
    sources = []
    scores = []
    vector_scores = []
    lexical_scores = []

    for chunk, score, metadata in retrieved_chunks:
        scores.append(score)
        vector_scores.append(metadata.get('vector_score', 0.0))
        lexical_scores.append(metadata.get('lexical_score', 0.0))
        sources.append({
            'doc_id': metadata.get('doc_id', 'unknown'),
            'doc_label': metadata.get('doc_label', metadata.get('doc_id', 'unknown')),
            'page': metadata.get('page', 'N/A'),
            'chunk_index': metadata.get('chunk_index', 'N/A'),
            'chunk_id': metadata.get('chunk_id', ''),
            'fragment': (chunk[:100] + '...') if len(chunk) > 100 else chunk,
            'score': round(score, 2),
            'vector_score': round(metadata.get('vector_score', 0.0), 2),
            'lexical_score': round(metadata.get('lexical_score', 0.0), 2),
            'matiere': metadata.get('matiere', ''),
            'enseignant': metadata.get('enseignant', ''),
        })

    avg_score = sum(scores) / len(scores) if scores else 0.0
    top1 = max(scores) if scores else 0.0
    top1_vector = max(vector_scores) if vector_scores else 0.0
    top1_lexical = max(lexical_scores) if lexical_scores else 0.0

    output = {
        'answer': answer,
        'sources': sources,
        'confidence': round(avg_score, 2),
        'reason_brief': f"Réponse basée sur {len(sources)} source(s), score moyen {avg_score:.2f}",
        'metadata_used': metadata_filter_used or {},
        'retrieval_stats': {
            'top1': round(top1, 2),
            'avg_topk': round(avg_score, 2),
            'threshold': retrieval_threshold,
            'k': len(retrieved_chunks),
            'vector_top1': round(top1_vector, 2),
            'lexical_top1': round(top1_lexical, 2),
        },
        'follow_ups': [],
    }

    return json.dumps(output, indent=2, ensure_ascii=False)
