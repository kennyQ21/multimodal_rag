"""
Multimodal RAG System — Entry Point
Run with: python run.py
or:        uvicorn app.api.main:app --reload
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
