-- Create training_doc table for storing document metadata
CREATE TABLE IF NOT EXISTS training_doc (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    doc_name VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size BIGINT NOT NULL,
    s3_url TEXT NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_status VARCHAR(50) DEFAULT 'pending',
    chunk_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_training_doc_user_id ON training_doc(user_id);
CREATE INDEX IF NOT EXISTS idx_training_doc_status ON training_doc(processing_status);
CREATE INDEX IF NOT EXISTS idx_training_doc_upload_date ON training_doc(upload_date);

-- Create document_chunks table for storing chunk metadata
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    doc_id INTEGER REFERENCES training_doc(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_metadata JSONB,
    vector_id VARCHAR(255), -- Milvus vector ID
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for chunks
CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id ON document_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_vector_id ON document_chunks(vector_id);
