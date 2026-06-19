"""
Hybrid Retrieval — BM25 + pgvector with Reciprocal Rank Fusion (RRF).

Pipeline:
    BM25  top-50
    pgvector top-30
     └── RRF → top-60 merged
"""
import logging
from typing import Optional
from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)

# ── BM25 Singleton ─────────────────────────────────────────────────────────────
_bm25: Optional[BM25Okapi] = None
_bm25_chunk_ids: list[str] = []


def _load_bm25():
    global _bm25, _bm25_chunk_ids
    from app.storage.database import SessionLocal
    from app.storage.models import Chunk

    db = SessionLocal()
    try:
        chunks = db.query(Chunk.id, Chunk.retrieval_text).all()
        _bm25_chunk_ids = [c.id for c in chunks]
        tokenized = [
            (c.retrieval_text or "").lower().split()
            for c in chunks
        ]
        _bm25 = BM25Okapi(tokenized) if tokenized else None
        logger.info(f"BM25 index built with {len(_bm25_chunk_ids)} chunks.")
    finally:
        db.close()


def reload_bm25():
    """Call this after ingestion to refresh the BM25 index."""
    global _bm25, _bm25_chunk_ids
    _bm25 = None
    _bm25_chunk_ids = []
    _load_bm25()


def _bm25_search(query: str, top_k: int) -> list[tuple[str, float]]:
    global _bm25, _bm25_chunk_ids
    if _bm25 is None:
        _load_bm25()
    if _bm25 is None or not _bm25_chunk_ids:
        return []

    tokens = query.lower().split()
    scores = _bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(str(_bm25_chunk_ids[i]), float(scores[i])) for i in top_idx if scores[i] > 0]


# ── ChromaDB Search ────────────────────────────────────────────────────────────

def _vector_search(query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
    try:
        from app.storage.chroma import search_chroma
        return search_chroma(query_embedding, top_k)
    except Exception as e:
        logger.error(f"ChromaDB search error: {e}")
        return []

# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: list[list[tuple[str, float]]],
    k: int = 60,
    top_k: int = 60,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists via RRF."""
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, (chunk_id, _) in enumerate(results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ── Public API ─────────────────────────────────────────────────────────────────

def hybrid_search(query: str, queries: list[str] = None, top_k: int = 30) -> list[tuple[str, float]]:
    """BM25-only search (simplified for stability)."""
    search_queries = [query]
    if queries:
        search_queries.extend(queries)

    # BM25 Search
    bm25_results = []
    for q in set(search_queries):
        bm25_results.extend(_bm25_search(q, top_k=top_k))

    # Aggregate scores for duplicate chunk_ids
    aggregated = {}
    for chunk_id, score in bm25_results:
        aggregated[chunk_id] = max(aggregated.get(chunk_id, 0.0), score)

    sorted_results = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]
