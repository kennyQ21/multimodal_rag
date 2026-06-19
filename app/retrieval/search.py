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


# ── pgvector Search ────────────────────────────────────────────────────────────

def _vector_search(query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
    from app.storage.database import SessionLocal
    from app.storage.models import Chunk
    from sqlalchemy import text

    db = SessionLocal()
    try:
        # Use pgvector operator for cosine distance — smaller = more similar
        vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        rows = db.execute(
            text(
                """
                SELECT id, 1 - (embedding <=> :vec::vector) AS similarity
                FROM chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> :vec::vector
                LIMIT :k
                """
            ),
            {"vec": vec_str, "k": top_k},
        ).fetchall()
        return [(str(r[0]), float(r[1])) for r in rows]
    except Exception as e:
        logger.error(f"pgvector search error: {e}")
        return []
    finally:
        db.close()


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

def hybrid_search(query: str, queries: list[str] | None = None) -> list[tuple[str, float]]:
    """
    Run hybrid BM25 + vector search for one or more query strings.
    Returns merged (chunk_id, rrf_score) list, top-60.
    """
    from app.config import get_settings
    from app.retrieval.embeddings import embed_texts

    settings = get_settings()
    all_queries = queries or [query]

    bm25_lists, vector_lists = [], []
    for q in all_queries:
        bm25_lists.append(_bm25_search(q, settings.bm25_top_k))
        emb = embed_texts([q])[0]
        vector_lists.append(_vector_search(emb, settings.vector_top_k))

    merged = reciprocal_rank_fusion(bm25_lists + vector_lists, k=settings.rrf_k, top_k=60)
    return merged
