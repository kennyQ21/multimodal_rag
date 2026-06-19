"""
FastAPI Application — Multimodal RAG System

Endpoints:
    GET  /health
    GET  /metrics
    POST /ingest
    POST /query
    POST /reindex
"""
import os
import shutil
import time
import logging
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from app.config import get_settings
from app.storage.database import SessionLocal, engine
from app.storage.models import Base, QueryLog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# ── Create all tables on startup ───────────────────────────────────────────────
def _init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")


app = FastAPI(
    title="Multimodal RAG — Technical Manual Q&A",
    description="Answers how-to questions from image-heavy manuals with step-by-step instructions and citations.",
    version="1.0.0",
)


# ── Startup event: warm up all models ─────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    logger.info("=== Multimodal RAG startup ===")
    _init_db()

    logger.info("Warming up embedding model…")
    try:
        from app.retrieval.embeddings import get_embedding_model
        get_embedding_model()
    except Exception as e:
        logger.warning(f"Embedding model warmup failed (will retry on first use): {e}")

    logger.info("Warming up reranker…")
    try:
        from app.reranker.model import get_reranker
        get_reranker()
    except Exception as e:
        logger.warning(f"Reranker warmup failed (will retry on first use): {e}")

    logger.info("Checking OCR availability…")
    try:
        from app.ingestion.ocr import get_ocr
        get_ocr()
    except Exception as e:
        logger.warning(f"OCR warmup failed (will use fallback): {e}")

    logger.info("Loading BM25 index…")
    try:
        from app.retrieval.search import reload_bm25
        reload_bm25()
    except Exception as e:
        logger.warning(f"BM25 load failed: {e}")

    logger.info("=== Startup complete ===")


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    steps: list[dict]
    citations: list[dict]
    total_steps: int
    latency_sec: float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/metrics", tags=["System"])
def metrics():
    db = SessionLocal()
    try:
        query_count = db.query(QueryLog).count()
        from app.storage.models import Chunk, Document, Image
        chunk_count = db.query(Chunk).count()
        doc_count = db.query(Document).count()
        img_count = db.query(Image).count()
        return {
            "total_queries": query_count,
            "total_documents": doc_count,
            "total_chunks": chunk_count,
            "total_images": img_count,
        }
    finally:
        db.close()


@app.post("/ingest", tags=["Ingestion"])
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF manual and ingest it into the RAG system."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    os.makedirs(settings.data_dir, exist_ok=True)
    save_path = os.path.join(settings.data_dir, file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"Saved uploaded file to {save_path}")

    from app.ingestion.pipeline import process_pdf
    doc_id = process_pdf(save_path)

    # Refresh BM25 index after ingestion
    from app.retrieval.search import reload_bm25
    reload_bm25()

    return {"status": "success", "document_id": doc_id, "file": file.filename}


@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query(request: QueryRequest):
    """Ask a how-to question against the ingested manual."""
    from app.graph.workflow import build_graph

    start = time.perf_counter()

    graph = build_graph()
    result = graph.invoke({
        "query": request.question,
        "expanded_queries": [],
        "retrieved_ids": [],
        "ranked_chunks": [],
        "expanded_chunks": [],
        "groq_answer": "",
        "steps": [],
        "citations": [],
        "final_response": {},
    })

    elapsed = round(time.perf_counter() - start, 3)

    final = result.get("final_response", {})

    # Log query
    db = SessionLocal()
    try:
        db.add(QueryLog(
            question=request.question,
            latency_sec=elapsed,
            result_count=len(final.get("steps", [])),
        ))
        db.commit()
    finally:
        db.close()

    return QueryResponse(
        query=final.get("query", request.question),
        answer=final.get("answer", ""),
        steps=final.get("steps", []),
        citations=final.get("citations", []),
        total_steps=final.get("total_steps", 0),
        latency_sec=elapsed,
    )


@app.post("/reindex", tags=["System"])
def reindex():
    """Rebuild the BM25 in-memory index from the current database state."""
    from app.retrieval.search import reload_bm25
    reload_bm25()
    return {"status": "BM25 index reloaded."}
