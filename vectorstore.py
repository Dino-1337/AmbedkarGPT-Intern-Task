# vectorstore.py
import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"
import chromadb
import numpy as np
from config import CHROMA_DIR, TOP_K
from typing import List


# ----------------------------------------------------------
# Chroma client (local mode) - UPDATED FOR NEW CHROMA VERSION
# ----------------------------------------------------------
try:
    # New Chroma client initialization
    client = chromadb.PersistentClient(path=CHROMA_DIR)
except Exception as e:
    print(f"Error initializing Chroma client: {e}")
    # Fallback to ephemeral client if persistent fails
    client = chromadb.EphemeralClient()


# ----------------------------------------------------------
# Create or load collection
# ----------------------------------------------------------
def get_collection():
    """Local Chroma collection guaranteed create/load ho jayega"""
    return client.get_or_create_collection(
        name="rag_store",
        metadata={"hnsw:space": "cosine"}  # cosine similarity best for MiniLM
    )


# ----------------------------------------------------------
# Build vectorstore (called after chunking + embedding)
# ----------------------------------------------------------
def build_vectorstore(chunks: List[str], embeddings: List[List[float]]):
    """Chunks + embeddings ko Chroma mein store karta hai"""
    collection = get_collection()

    # IDs generate karna zaroori hota hai
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Purane index clean karna (Assignment 1 mein ek hi document hoga)
    clear_collection(collection)

    # Ensure embeddings are lists, not numpy arrays
    embeddings_as_lists = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]

    # Add to Chroma
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings_as_lists
    )


# ----------------------------------------------------------
# Retrieval wrapper
# ----------------------------------------------------------
def retrieve(query_embedding, top_k=TOP_K):
    """Query embedding se top-k chunks retrieve karta hai"""
    collection = get_collection()

    # Convert query embedding to list if it's a numpy array
    if hasattr(query_embedding, 'tolist'):
        query_embedding = query_embedding.tolist()
    
    # Ensure it's a list of floats, not a nested list
    if isinstance(query_embedding, list) and len(query_embedding) > 0 and isinstance(query_embedding[0], list):
        query_embedding = query_embedding[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Agar koi result nahi मिला
    if not results or len(results["documents"]) == 0:
        return []

    hits = []
    docs = results["documents"][0]
    ids = results["ids"][0]
    dists = results["distances"][0]  # cosine distance

    for i in range(len(docs)):
        hits.append({
            "id": ids[i],
            "content": docs[i],
            "score": float(dists[i])  # distance = similarity ka opposite
        })

    return hits


# ----------------------------------------------------------
# Utility: Clear old vectors
# ----------------------------------------------------------
def clear_collection(collection):
    """Purane vectors remove kar deta hai (fresh build ke liye)"""
    try:
        all_items = collection.get()
        if all_items and "ids" in all_items and all_items["ids"]:
            collection.delete(all_items["ids"])
    except:
        pass  # Agar empty hai toh koi issue nahi