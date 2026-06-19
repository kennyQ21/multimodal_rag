"""
LangGraph Workflow — Multimodal RAG

Flow:
    User Query
    ↓ intent
    ↓ query_expansion     (Groq LLM)
    ↓ hybrid_retrieval    (BM25 + pgvector + RRF)
    ↓ rerank              (bge-reranker-v2-m3)
    ↓ neighbor_expansion  (pull sibling steps)
    ↓ context_builder     (assemble text for Groq)
    ↓ groq_generation     (Groq llama-3.3-70b — answer composition only)
    ↓ citation_formatter  (attach images + citations)
    ↓ END

Groq is used ONLY for:
    - Query expansion
    - Answer composition / step ordering
Groq does NOT do: OCR, chunking, retrieval, embeddings.
"""
import logging
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    query: str
    expanded_queries: list[str]
    retrieved_ids: list[str]            # chunk IDs from RRF
    ranked_chunks: list[dict]           # after reranking
    expanded_chunks: list[dict]         # after neighbor expansion
    groq_answer: str                    # raw Groq prose answer
    steps: list[dict]                   # final structured steps
    citations: list[dict]
    final_response: dict


# ── Node 1: Intent (pass-through, extensible) ──────────────────────────────────

def intent_node(state: RAGState) -> RAGState:
    logger.debug("intent_node: pass-through")
    return state


# ── Node 2: Query Expansion via Groq ──────────────────────────────────────────

def query_expansion_node(state: RAGState) -> RAGState:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
    from app.config import get_settings

    settings = get_settings()
    query = state["query"]

    try:
        llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            groq_api_key=settings.groq_api_key,
        )
        prompt = (
            f"You are a technical document retrieval assistant.\n"
            f"Original query: '{query}'\n"
            f"Generate exactly 2 alternative phrasings of this query to improve document retrieval. "
            f"Return ONLY the 2 alternatives, one per line, no numbering, no explanations."
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        extras = [l.strip() for l in response.content.strip().split("\n") if l.strip()][:2]
        state["expanded_queries"] = [query] + extras
        logger.debug(f"Expanded queries: {state['expanded_queries']}")
    except Exception as e:
        logger.warning(f"Query expansion failed: {e} — using original query only.")
        state["expanded_queries"] = [query]

    return state


# ── Node 3: Hybrid Retrieval ───────────────────────────────────────────────────

def hybrid_retrieval_node(state: RAGState) -> RAGState:
    from app.retrieval.search import hybrid_search

    results = hybrid_search(
        query=state["query"],
        queries=state.get("expanded_queries"),
    )
    state["retrieved_ids"] = [chunk_id for chunk_id, _ in results]
    logger.debug(f"Hybrid retrieval: {len(state['retrieved_ids'])} candidates.")
    return state




# ── Node 5: Neighbor Expansion ─────────────────────────────────────────────────

def neighbor_expansion_node(state: RAGState) -> RAGState:
    """
    For each top-ranked chunk, fetch ALL chunks in the same procedure
    so the full step sequence is available — never partial procedures.
    """
    from app.storage.database import SessionLocal
    from app.storage.models import Chunk

    ids = state.get("retrieved_ids", [])
    if not ids:
        state["expanded_chunks"] = []
        return state

    db = SessionLocal()
    try:
        # We simulate "ranked" by fetching the top 8 (rerank_top_k) from retrieved_ids directly
        from app.config import get_settings
        settings = get_settings()
        top_ids = ids[:settings.rerank_top_k]
        
        rows = db.query(Chunk).filter(Chunk.id.in_(top_ids)).all()
        ranked = [
            {"chunk_id": r.id, "retrieval_text": r.retrieval_text or "",
             "steps": r.steps, "page_start": r.page_start,
             "page_end": r.page_end, "procedure_title": r.procedure_title,
             "procedure_id": r.procedure_id}
            for r in rows
        ]
        
        proc_ids = list({c["procedure_id"] for c in ranked if c.get("procedure_id")})

        rows = (
            db.query(Chunk)
            .filter(Chunk.procedure_id.in_(proc_ids))
            .order_by(Chunk.page_start)
            .all()
        )
        state["expanded_chunks"] = [
            {"chunk_id": r.id, "retrieval_text": r.retrieval_text or "",
             "steps": r.steps, "page_start": r.page_start,
             "page_end": r.page_end, "procedure_title": r.procedure_title,
             "procedure_id": r.procedure_id}
            for r in rows
        ]
    finally:
        db.close()

    logger.debug(f"Neighbor expansion: {len(state['expanded_chunks'])} total chunks.")
    return state


# ── Node 6: Context Builder ────────────────────────────────────────────────────

def context_builder_node(state: RAGState) -> RAGState:
    """
    Assemble a structured text context from expanded chunks
    to feed to Groq for answer composition.
    """
    chunks = state.get("expanded_chunks", [])
    lines = []
    for chunk in chunks:
        title = chunk.get("procedure_title", "")
        lines.append(f"\n### {title}")
        for step in chunk.get("steps", []):
            step_num = step.get("step", "?")
            instruction = step.get("instruction", "")
            if instruction:
                lines.append(f"Step {step_num}: {instruction}")

    state["groq_answer"] = "\n".join(lines)   # will be replaced by Groq
    return state


# ── Node 7: Groq Generation ────────────────────────────────────────────────────

def groq_generation_node(state: RAGState) -> RAGState:
    """
    Groq receives the retrieved context and the user query.
    Groq ONLY does:
        - Procedure formatting / ordering
        - Answer composition using ONLY retrieved content
    Groq does NOT hallucinate steps — it must cite only what is in context.
    """
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    from app.config import get_settings

    settings = get_settings()
    context = state.get("groq_answer", "")
    query = state["query"]

    if not context.strip():
        state["groq_answer"] = "No relevant steps found in the manual."
        return state

    system_prompt = (
        "You are a technical manual assistant. "
        "You will be given extracted steps from a technical manual and a user question. "
        "Your job is to select and format ONLY the steps relevant to the question, "
        "preserving their exact order from the manual. "
        "Do NOT add, invent, or paraphrase steps — only use what is provided. "
        "Format as numbered steps. If a step has associated images, note [See figure]. "
        "Be concise and precise."
    )
    user_prompt = (
        f"User question: {query}\n\n"
        f"Extracted manual content:\n{context}\n\n"
        "Please provide the relevant how-to steps in order."
    )

    try:
        llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            groq_api_key=settings.groq_api_key,
            max_tokens=2048,
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        state["groq_answer"] = response.content.strip()
        logger.debug("Groq generation complete.")
    except Exception as e:
        logger.error(f"Groq generation failed: {e}")
        # Fall back to raw extracted context if Groq fails
        state["groq_answer"] = context

    return state


# ── Node 8: Citation Formatter ─────────────────────────────────────────────────

def citation_formatter_node(state: RAGState) -> RAGState:
    """
    Build the final structured response with steps, images, and citations.
    """
    from app.storage.database import SessionLocal
    from app.storage.models import Image

    db = SessionLocal()
    try:
        chunks = state.get("expanded_chunks", [])
        formatted_steps = []
        citations = []

        for chunk in chunks:
            for step in chunk.get("steps", []):
                instruction = step.get("instruction", "")
                step_images = []

                for img_id in step.get("images", []):
                    img_row = db.query(Image).filter(Image.id == img_id).first()
                    if img_row:
                        step_images.append({
                            "image_id": img_id,
                            "path": img_row.path,
                            "figure_label": img_row.figure_label or "",
                            "caption": img_row.caption or "",
                            "citation": f"Source: Page {img_row.page_number} · {img_row.figure_label or 'Figure'}",
                        })
                        citations.append({
                            "page": img_row.page_number,
                            "figure": img_row.figure_label or "",
                            "chunk_id": chunk["chunk_id"],
                            "image_id": img_id,
                        })

                if instruction or step_images:
                    formatted_steps.append({
                        "step": step.get("step"),
                        "instruction": instruction,
                        "images": step_images,
                        "page": chunk["page_start"],
                        "procedure": chunk.get("procedure_title", ""),
                    })

        state["steps"] = formatted_steps
        state["citations"] = citations
        state["final_response"] = {
            "query": state["query"],
            "answer": state.get("groq_answer", ""),
            "steps": formatted_steps,
            "citations": citations,
            "total_steps": len(formatted_steps),
        }
    finally:
        db.close()

    return state


# ── Build Graph ────────────────────────────────────────────────────────────────

def build_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("intent", intent_node)
    workflow.add_node("query_expansion", query_expansion_node)
    workflow.add_node("hybrid_retrieval", hybrid_retrieval_node)
    workflow.add_node("neighbor_expansion", neighbor_expansion_node)
    workflow.add_node("context_builder", context_builder_node)
    workflow.add_node("groq_generation", groq_generation_node)
    workflow.add_node("citation_formatter", citation_formatter_node)

    workflow.set_entry_point("intent")
    workflow.add_edge("intent", "query_expansion")
    workflow.add_edge("query_expansion", "hybrid_retrieval")
    workflow.add_edge("hybrid_retrieval", "neighbor_expansion")
    workflow.add_edge("neighbor_expansion", "context_builder")
    workflow.add_edge("context_builder", "groq_generation")
    workflow.add_edge("groq_generation", "citation_formatter")
    workflow.add_edge("citation_formatter", END)

    return workflow.compile()
