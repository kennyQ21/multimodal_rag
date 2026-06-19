from app.retrieval.search import reciprocal_rank_fusion

def test_reciprocal_rank_fusion():
    bm25 = [("chunk1", 2.0), ("chunk2", 1.5)]
    vector = [("chunk2", 0.9), ("chunk3", 0.8)]
    
    rrf = reciprocal_rank_fusion(bm25, vector, k=60, top_k=2)
    
    # chunk2 is rank 2 in bm25 and rank 1 in vector
    # chunk1 is rank 1 in bm25 and not in vector
    # 1/(60+2) + 1/(60+1) > 1/(60+1) + 0
    assert rrf[0][0] == "chunk2"
    assert rrf[1][0] == "chunk1"
