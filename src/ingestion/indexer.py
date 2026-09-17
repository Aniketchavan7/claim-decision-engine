import os
import json
import logging
from typing import Tuple, List, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import bm25s

from src.models.evidence import PolicyChunk
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.chunker import chunk_document

logger = logging.getLogger(__name__)

def build_index(chunks: List[PolicyChunk], index_dir: str) -> None:
    logger.info(f"Building index in {index_dir}")
    os.makedirs(index_dir, exist_ok=True)
    
    # 1. Save chunks to chunks.json
    chunks_data = [chunk.model_dump() for chunk in chunks]
    with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2)
        
    # 2. Build FAISS index
    logger.info("Building FAISS index...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    
    # 3. Build BM25 index
    logger.info("Building BM25 index...")
    import Stemmer
    stemmer = Stemmer.Stemmer("english")
    corpus = [c.text for c in chunks]
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
    
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    bm25_dir = os.path.join(index_dir, "bm25")
    os.makedirs(bm25_dir, exist_ok=True)
    retriever.save(bm25_dir)
    
    logger.info("Indexing complete.")

def load_or_build(pdf_path: str, index_dir: str) -> Tuple[List[PolicyChunk], Any, Any]:
    chunks_path = os.path.join(index_dir, "chunks.json")
    faiss_path = os.path.join(index_dir, "faiss.index")
    bm25_dir = os.path.join(index_dir, "bm25")
    
    if os.path.exists(chunks_path) and os.path.exists(faiss_path) and os.path.exists(bm25_dir):
        logger.info("Loading existing indexes...")
        
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
            chunks = [PolicyChunk.model_validate(c) for c in chunks_data]
            
        faiss_index = faiss.read_index(faiss_path)
        
        bm25_retriever = bm25s.BM25.load(bm25_dir, load_corpus=False)
        
        return chunks, faiss_index, bm25_retriever
    else:
        logger.info("Indexes not found. Building from scratch...")
        blocks = parse_pdf(pdf_path)
        chunks = chunk_document(blocks)
        build_index(chunks, index_dir)
        
        return load_or_build(pdf_path, index_dir)
