import chromadb
from typing import List, Any
from config import CHROMA_DIR, TOP_K

# Try persistent Chroma, fallback to ephemeral
try:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
except Exception:
    client = chromadb.EphemeralClient()


def get_collection():
    """Local Chroma collection (create if not exists)"""
    return client.get_or_create_collection(
        name="rag_store",
        metadata={"hnsw:space": "cosine"}
    )


def build_vectorstore(chunks: List[str], embeddings: List[Any]):
    """Store chunks and embeddings in Chroma; replace any existing index"""
    coll = get_collection()
    clear_collection(coll)

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Ensure embeddings are plain nested lists (not numpy)
    emb_lists = []
    for e in embeddings:
        if hasattr(e, "tolist"):
            emb_lists.append(e.tolist())
        else:
            emb_lists.append(e)

    coll.add(
        ids=ids,
        documents=chunks,
        embeddings=emb_lists
    )


def retrieve(query_embedding, top_k: int = TOP_K):
    """Return list of hits: {id, content, score} where score is similarity (0..1)"""
    coll = get_collection()

    # normalize input to plain list
    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()
    if isinstance(query_embedding, list) and len(query_embedding) > 0 and isinstance(query_embedding[0], list):
        query_embedding = query_embedding[0]

    results = coll.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    if not results or not results.get("documents"):
        return []

    docs = results["documents"][0]
    ids = results.get("ids", [[]])[0]
    dists = results.get("distances", [[]])[0]  # lower = closer if using distance

    hits = []
    for i in range(len(docs)):
        # Convert distance -> similarity (simple linear transform)
        try:
            dist = float(dists[i])
            sim = max(0.0, 1.0 - dist)
        except Exception:
            sim = 0.0

        hits.append({
            "id": ids[i] if i < len(ids) else f"chunk_{i}",
            "content": docs[i],
            "score": float(sim)
        })

    return hits


def clear_collection(collection):
    """Delete all items in collection"""
    try:
        data = collection.get()
        if data and "ids" in data and data["ids"]:
            collection.delete(ids=data["ids"])
    except Exception:
        pass
