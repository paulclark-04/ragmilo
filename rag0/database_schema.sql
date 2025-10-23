-- ECE Paris RAG Database Schema
-- SQLite database for file management and classification

-- Table for storing file metadata and classifications
CREATE TABLE files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL UNIQUE,
    file_size INTEGER,
    file_type TEXT, -- 'pdf', 'docx', 'txt', etc.
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_modified TIMESTAMP,
    
    -- Classification fields
    matiere TEXT NOT NULL,
    enseignant TEXT NOT NULL,
    semestre TEXT NOT NULL,
    promo TEXT NOT NULL,
    
    -- Additional metadata
    doc_id TEXT UNIQUE, -- Generated ID for RAG system
    doc_label TEXT, -- Human-readable label
    description TEXT,
    tags TEXT, -- JSON array of tags
    
    -- Processing status
    is_processed BOOLEAN DEFAULT FALSE,
    processing_date TIMESTAMP,
    chunk_count INTEGER DEFAULT 0,
    
    -- File integrity
    file_hash TEXT, -- SHA256 hash for duplicate detection
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing RAG chunks with references to files
CREATE TABLE rag_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    chunk_id TEXT NOT NULL UNIQUE, -- Format: docid:page:index
    chunk_text TEXT NOT NULL,
    page_number INTEGER,
    chunk_index INTEGER,
    
    -- Embedding and search data
    embedding BLOB, -- Store embedding as binary data
    embedding_model TEXT,
    
    -- Metadata for filtering
    matiere TEXT NOT NULL,
    enseignant TEXT NOT NULL,
    semestre TEXT NOT NULL,
    promo TEXT NOT NULL,
    
    -- Search scores (for caching)
    last_vector_score REAL,
    last_lexical_score REAL,
    last_combined_score REAL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (file_id) REFERENCES files (id) ON DELETE CASCADE
);

-- Table for storing search history and analytics
CREATE TABLE search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    filters TEXT, -- JSON of applied filters
    results_count INTEGER,
    response_time_ms INTEGER,
    user_feedback INTEGER, -- 1-5 rating
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing system configuration
CREATE TABLE system_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_files_classification ON files(matiere, enseignant, semestre, promo);
CREATE INDEX idx_files_processed ON files(is_processed);
CREATE INDEX idx_chunks_file_id ON rag_chunks(file_id);
CREATE INDEX idx_chunks_classification ON rag_chunks(matiere, enseignant, semestre, promo);
CREATE INDEX idx_chunks_chunk_id ON rag_chunks(chunk_id);
CREATE INDEX idx_search_history_timestamp ON search_history(timestamp);

-- Views for common queries
CREATE VIEW file_summary AS
SELECT 
    f.id,
    f.filename,
    f.matiere,
    f.enseignant,
    f.semestre,
    f.promo,
    f.is_processed,
    f.chunk_count,
    f.upload_date,
    COUNT(c.id) as actual_chunks
FROM files f
LEFT JOIN rag_chunks c ON f.id = c.file_id
GROUP BY f.id, f.filename, f.matiere, f.enseignant, f.semestre, f.promo, f.is_processed, f.chunk_count, f.upload_date;

-- Insert default configuration
INSERT INTO system_config (key, value, description) VALUES
('default_embedding_model', 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf', 'Default embedding model for new files'),
('chunk_size', '500', 'Default chunk size for text processing'),
('chunk_overlap', '50', 'Default overlap between chunks'),
('vector_threshold', '0.35', 'Default similarity threshold for vector search'),
('hybrid_alpha', '0.65', 'Default alpha for hybrid search combination');
