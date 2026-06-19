import os
import shutil
import time
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from app.ingestion.pipeline import process_pdf
from app.graph.workflow import build_graph
from app.storage.database import engine, Base

# Create tables
# In pgvector we need to ensure the extension exists, this is usually handled via SQL or setup.
# We will execute CREATE EXTENSION IF NOT EXISTS vector in the DB if needed.
# Since we use ankane/pgvector image, we can just create the tables.
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Multimodal RAG API")

class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():
    from sqlalchemy import text
    from app.storage.database import SessionLocal
    db = SessionLocal()
    db.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    db.commit()
    db.close()
    
    # Load singleton models to warm them up
    from app.ingestion.ocr import get_ocr_model
    from app.retrieval.embeddings import get_embedding_model
    from app.reranker.model import get_reranker_model
    from app.retrieval.search import load_bm25_index
    from app.ingestion.pipeline import get_lp_model
    
    get_ocr_model()
    get_embedding_model()
    get_reranker_model()
    get_lp_model()
    load_bm25_index()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/metrics")
def get_metrics():
    return {"message": "Metrics endpoint. Prometheus/Grafana could be attached here."}

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    doc_id = process_pdf(file_path)
    
    from app.retrieval.search import load_bm25_index
    load_bm25_index()
    
    return {"status": "success", "document_id": doc_id}

@app.post("/query")
def query_rag(request: QueryRequest):
    graph = build_graph()
    
    start_time = time.time()
    result = graph.invoke({
        "query": request.question,
        "expanded_queries": [],
        "retrieved_ids": [],
        "steps": [],
        "citations": [],
        "final_response": {}
    })
    end_time = time.time()
    
    response = result.get("final_response", {})
    response["latency_sec"] = end_time - start_time
    return response

@app.post("/reindex")
def reindex():
    from app.retrieval.search import load_bm25_index
    load_bm25_index()
    return {"status": "Reindexed successfully"}
