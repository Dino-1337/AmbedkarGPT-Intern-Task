# pipeline.py

import nltk
nltk.download('punkt', quiet=True)

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
import os

# Embedding model ko ek baar load karna best practice hota hai
embedder = SentenceTransformer(EMBEDDING_MODEL)


# ----------------------------------------------------------
# 1) Text Loader
# ----------------------------------------------------------
def load_text(file_path: str) -> str:
    """speech.txt load karne ka simple function"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File nahi mila: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ----------------------------------------------------------
# 2) Chunker (sentence-aware chunking)
# ----------------------------------------------------------
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Sentence-aware chunking taaki context cut na ho beech mein"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Agar sentence add karne se chunk bada ho jata hai → new chunk
        if len(current_chunk) + len(sentence) > chunk_size:
            chunks.append(current_chunk.strip())
            
            # Overlap ke liye last 'overlap' characters ko rakhte hain
            current_chunk = current_chunk[-overlap:] if overlap > 0 else ""

        current_chunk += " " + sentence

    # Last chunk add
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ----------------------------------------------------------
# 3) Embedding functions
# ----------------------------------------------------------
def embed_chunks(chunks: list):
    """Chunks ko embeddings mein convert karta hai"""
    return embedder.encode(chunks, normalize_embeddings=True)


def embed_query(query: str):
    """User query ki embedding"""
    return embedder.encode([query], normalize_embeddings=True)[0]
