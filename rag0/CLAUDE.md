ECE PARIS RAG PROJECT - CONTEXT MEMORY

PROJECT OVERVIEW
 Building a Retrieval Augmented Generation (RAG) system for ECE Paris engineering school to answer student questions using course materials and Milo notes. The system indexes documents by subject (matière), timeline (year/semester/session), and instructor.

DESIGN PRINCIPLES
 1. **Minimalism**: Keep code simple, sober, concise - avoid over-engineering
 2. **No hallucination**: Only answer from sources; say "info not found" if missing
 3. **Traceability**: Always cite sources with doc_id, page, score
 4. **Metadata-driven**: Filter by matière/enseignant/semester for precise retrieval
 5. **Local-first**: All processing runs locally, no cloud dependencies


KEY TECHNICAL DECISIONS
 - **Chunking Strategy**: 500 words with 50-word overlap (simple fixed-size)
 - **Retrieval Method**: Cosine similarity on embeddings, top-3 results
 - **Storage Format**: JSON for simplicity (will migrate to FAISS if >500 chunks)
 - **Embedding Distance**: Cosine similarity (standard for semantic search)

RAG PIPELINE FLOW
 1. **Ingestion**: PDF → PyMuPDF parse → chunk_text() → Ollama embed → vector_db.json
 2. **Query**: User question → Ollama embed → cosine similarity search → filter by metadata
 3. **Retrieval**: Top-3 chunks with scores + metadata
 4. **Generation**: Inject chunks as context → Ollama LLM → structured response
 5. **Output**: JSON format via output_formatter.py with sources and confidence

IMPORTANT CONSTRAINTS
 - Must integrate with Milo architecture (student notes platform)
 - Primary language: French (courses and responses in French)
 - Target users: ECE Paris engineering students
 - Privacy requirement: No personal data in outputs
 - Zero hallucination tolerance: Pedagogical accuracy is critical
 - Deployment: Must run on student laptops (local execution)

REFERENCE TUTORIAL
 Following HuggingFace tutorial: https://huggingface.co/blog/ngxson/make-your-own-rag
 - Simple, educational approach
 - No complex frameworks
 - Pure Python implementation
 - Easy to understand and modify

DEVELOPMENT PHILOSOPHY
 1. Start with simplest working implementation
 2. Iterate based on real evaluation metrics
 3. Avoid premature optimization
 4. Code must be readable by junior developers
 5. Document decisions clearly
 6. Test with real use cases before adding complexity

EVALUATION CRITERIA (Nov 7 Presentation)
 - System functionality: Does it retrieve and answer correctly?
 - Response quality: Accurate, well-cited, pedagogical
 - Integration with Milo: Works in target architecture
 - Code quality: Simple, maintainable, documented
 - Team understanding: All members can explain the system


SUCCESS METRICS
 - **Retrieval Precision@3**: >70% (top-3 results contain answer)
 - **Response Time**: <10 seconds per query
 - **Hallucination Rate**: 0% (must say "info not found" when appropriate)
 - **Citation Accuracy**: 100% (all sources properly attributed)
 - **Metadata Filtering**: 100% functional (filter by subject works)

KNOWN LIMITATIONS (Current MVP)
 - Only 11 chunks (small corpus)
 - No re-ranking (pure cosine similarity)
 - Simple fixed-size chunking (may lose semantic boundaries)
 - In-memory storage (will need FAISS for scale)
 - No query understanding (literal embedding match)
 - French language detection not implemented yet

FUTURE ENHANCEMENTS 
 - Hybrid search (BM25 + vector)
 - Semantic chunking (preserve context)
 - Query reformulation
 - Multi-hop reasoning
 - Milo notes integration
 - User feedback loop
 - A/B testing framework 