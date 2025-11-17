import nltk
nltk.download('punkt', quiet=True)

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
import os

# Embedding model ko ek baar load karna best practice hota hai
embedder = SentenceTransformer(EMBEDDING_MODEL)


def load_text(file_path: str) -> str:
    """Simple loader for speech.txt"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File nahi mila: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Sentence-aware chunking to avoid cutting sentences in middle"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-overlap:] if overlap > 0 else ""
        current_chunk += " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def embed_chunks(chunks: list):
    """Convert list of chunks to embeddings (normalized)"""
    emb = embedder.encode(chunks, normalize_embeddings=True)
    return emb.tolist()


def embed_query(query: str):
    """Embedding for a single query"""
    emb = embedder.encode([query], normalize_embeddings=True)[0]
    return emb.tolist()
