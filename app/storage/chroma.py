import os
import chromadb
from chromadb.config import Settings
import logging

logger = logging.getLogger(__name__)

_chroma_client = None
_collection = None

def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        db_path = os.path.join(os.getcwd(), "chroma_db")
        os.makedirs(db_path, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=db_path)
        logger.info(f"Initialized ChromaDB client at {db_path}")
    return _chroma_client

def get_chunk_collection():
    global _collection
    client = get_chroma_client()
    if _collection is None:
        _collection = client.get_or_create_collection(
            name="chunks",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def add_chunk_to_chroma(chunk_id: str, text: str, embedding: list[float], metadata: dict):
    collection = get_chunk_collection()
    collection.upsert(
        ids=[chunk_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata]
    )

def search_chroma(embedding: list[float], top_k: int = 30) -> list[tuple[str, float]]:
    collection = get_chunk_collection()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["distances"]
    )
    if not results or not results["ids"] or not results["ids"][0]:
        return []
    
    # chromadb cosine distances are (1 - cosine_similarity).
    # We want similarity scores where higher is better for RRF.
    # Also, some metrics return Euclidean. We configured cosine, so distances are 1 - cos_sim.
    # Hence, score = 1 - distance.
    ids = results["ids"][0]
    distances = results["distances"][0]
    
    scored = []
    for chunk_id, dist in zip(ids, distances):
        score = 1.0 - dist
        scored.append((chunk_id, score))
    return scored
