from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from app.retrieval.search import get_bm25_scores, get_vector_scores, reciprocal_rank_fusion
from app.retrieval.embeddings import get_embedding_model
from app.reranker.model import get_reranker_model
from app.storage.database import SessionLocal
from app.storage.models import Chunk, Image as DBImage

class GraphState(TypedDict):
    query: str
    expanded_queries: List[str]
    retrieved_ids: List[str]
    steps: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    final_response: Dict[str, Any]

def intent_node(state: GraphState):
    # Pass through or intent classification logic
    return state

import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

def query_expansion_node(state: GraphState):
    query = state["query"]
    api_key = os.getenv("GROQ_API_KEY", "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")
    
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=api_key)
        prompt = f"Given the user query: '{query}', generate 2 alternative ways to ask this question for a retrieval system. Return ONLY the alternative queries, one per line."
        response = llm.invoke([HumanMessage(content=prompt)])
        expanded = [line.strip() for line in response.content.split('\n') if line.strip()]
        state["expanded_queries"] = [query] + expanded
    except Exception as e:
        print(f"LLM Error: {e}")
        state["expanded_queries"] = [query]
        
    return state

def hybrid_retrieval_node(state: GraphState):
    queries = state.get("expanded_queries", [state["query"]])
    
    all_bm25_results = []
    all_vector_results = []
    
    embedder = get_embedding_model()
    
    for q in queries:
        all_bm25_results.extend(get_bm25_scores(q, top_k=50))
        query_emb = embedder.encode(q).tolist()
        all_vector_results.extend(get_vector_scores(query_emb, top_k=30))
    
    # RRF
    rrf_results = reciprocal_rank_fusion(all_bm25_results, all_vector_results, k=60, top_k=60)
    
    state["retrieved_ids"] = [chunk_id for chunk_id, score in rrf_results]
    return state

def rerank_node(state: GraphState):
    query = state["query"]
    retrieved_ids = state["retrieved_ids"]
    
    if not retrieved_ids:
        return state
        
    db = SessionLocal()
    chunks = db.query(Chunk).filter(Chunk.id.in_(retrieved_ids)).all()
    chunk_map = {c.id: c for c in chunks}
    db.close()
    
    pairs = []
    valid_ids = []
    for cid in retrieved_ids:
        if cid in chunk_map:
            pairs.append([query, chunk_map[cid].retrieval_text])
            valid_ids.append(cid)
            
    reranker = get_reranker_model()
    scores = reranker.compute_score(pairs)
    
    # Sort by score descending
    scored_ids = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
    
    # Top 8
    final_top_8 = [cid for cid, score in scored_ids[:8]]
    state["retrieved_ids"] = final_top_8
    return state

def neighbor_expansion_node(state: GraphState):
    # For the top chunks, we might want to get the whole procedure or just pass them forward.
    # The spec says "Ordered Steps", so we order them by procedure_id and step_number, or just page_number.
    db = SessionLocal()
    retrieved_ids = state["retrieved_ids"]
    if not retrieved_ids:
        db.close()
        return state
        
    chunks = db.query(Chunk).filter(Chunk.id.in_(retrieved_ids)).all()
    
    # Group by procedure to return coherent steps
    procedures = set([c.procedure_id for c in chunks])
    
    # For now, let's just fetch all chunks for the top 1 or 2 procedures matched
    # to ensure complete step-by-step instructions.
    expanded_chunks = db.query(Chunk).filter(Chunk.procedure_id.in_(procedures)).order_by(Chunk.page_start).all()
    
    state["retrieved_ids"] = [c.id for c in expanded_chunks]
    db.close()
    return state

def step_composer_node(state: GraphState):
    db = SessionLocal()
    retrieved_ids = state["retrieved_ids"]
    chunks = db.query(Chunk).filter(Chunk.id.in_(retrieved_ids)).all()
    
    # Sort chunks by page/procedure
    chunks.sort(key=lambda x: x.page_start)
    
    steps = []
    citations = []
    for chunk in chunks:
        for step in chunk.steps:
            steps.append({
                "instruction": step["instruction"],
                "images": step["images"],
                "chunk_id": chunk.id,
                "page": chunk.page_start
            })
            for img_hash in step["images"]:
                citations.append({
                    "page": chunk.page_start,
                    "chunk_id": chunk.id,
                    "image_id": img_hash
                })
    
    state["steps"] = steps
    state["citations"] = citations
    db.close()
    return state

def citation_formatter_node(state: GraphState):
    db = SessionLocal()
    citations = state["citations"]
    steps = state["steps"]
    
    formatted_steps = []
    for step in steps:
        step_imgs = []
        for img_id in step["images"]:
            img_record = db.query(DBImage).filter(DBImage.id == img_id).first()
            if img_record:
                step_imgs.append({
                    "path": img_record.path,
                    "citation": f"Source: Page {img_record.page.page_number} \u00b7 Figure" if img_record.page else f"Source: Page {step['page']}"
                })
        
        formatted_steps.append({
            "instruction": step["instruction"],
            "images": step_imgs
        })
        
    state["final_response"] = {
        "steps": formatted_steps,
        "citations": citations
    }
    db.close()
    return state

def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("intent", intent_node)
    workflow.add_node("query_expansion", query_expansion_node)
    workflow.add_node("hybrid_retrieval", hybrid_retrieval_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("neighbor_expansion", neighbor_expansion_node)
    workflow.add_node("step_composer", step_composer_node)
    workflow.add_node("citation_formatter", citation_formatter_node)
    
    workflow.set_entry_point("intent")
    workflow.add_edge("intent", "query_expansion")
    workflow.add_edge("query_expansion", "hybrid_retrieval")
    workflow.add_edge("hybrid_retrieval", "rerank")
    workflow.add_edge("rerank", "neighbor_expansion")
    workflow.add_edge("neighbor_expansion", "step_composer")
    workflow.add_edge("step_composer", "citation_formatter")
    workflow.add_edge("citation_formatter", END)
    
    return workflow.compile()
