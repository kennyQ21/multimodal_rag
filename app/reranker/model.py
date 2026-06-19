"""
Reranker Singleton — BAAI/bge-reranker-v2-m3 (CPU).

Rules:
- Single instance per process.
- Loaded lazily.
- Weights cached in models/huggingface/.
- Only reranks final candidates (never full corpus).
"""
import logging
from app.config import get_settings

logger = logging.getLogger(__name__)

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker
        settings = get_settings()
        logger.info(f"Loading reranker: {settings.reranker_model}")
        _reranker = FlagReranker(
            settings.reranker_model,
            use_fp16=False,   # CPU — no FP16
        )
        logger.info("Reranker ready.")
    return _reranker


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """
    Rerank a list of candidate dicts using (query, retrieval_text) pairs.

    Each candidate must have:
        - 'chunk_id': str
        - 'retrieval_text': str

    Returns the same dicts sorted by reranker score (descending),
    with an added 'rerank_score' key.
    """
    if not candidates:
        return candidates

    reranker = get_reranker()
    settings = get_settings()

    pairs = [[query, c["retrieval_text"] or ""] for c in candidates]
    scores = reranker.compute_score(pairs, batch_size=settings.reranker_batch_size)

    # scores can be a single float if only 1 pair
    if isinstance(scores, float):
        scores = [scores]

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
