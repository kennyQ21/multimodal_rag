"""
Unit tests for hybrid retrieval — specifically RRF.
These do NOT require a database connection.
"""
from app.retrieval.search import reciprocal_rank_fusion


def test_rrf_merges_correctly():
    """chunk2 appears in both lists — should score highest."""
    bm25 = [("chunk1", 2.0), ("chunk2", 1.5)]
    vec  = [("chunk2", 0.9), ("chunk3", 0.8)]

    merged = reciprocal_rank_fusion([bm25, vec], k=60, top_k=3)
    ids = [m[0] for m in merged]

    # chunk2 is in both lists → highest RRF score
    assert ids[0] == "chunk2", f"Expected chunk2 first, got {ids[0]}"


def test_rrf_handles_empty_list():
    result = reciprocal_rank_fusion([], k=60, top_k=5)
    assert result == []


def test_rrf_single_list():
    single = [("a", 1.0), ("b", 0.5)]
    result = reciprocal_rank_fusion([single], k=60, top_k=2)
    assert result[0][0] == "a"
