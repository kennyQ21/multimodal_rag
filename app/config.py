from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
import os


class Settings(BaseSettings):
    # API Keys
    groq_api_key: str = "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"
    groq_model: str = "llama-3.3-70b-versatile"

    # Database
    database_url: str = Field(default="sqlite:///./multimodal_rag.db")

    # Model cache dirs
    hf_home: str = "models/huggingface"
    transformers_cache: str = "models/huggingface"
    paddleocr_home: str = "models/paddleocr"

    # Data directories
    data_dir: str = "data"
    pages_dir: str = "data/pages"
    crops_dir: str = "data/crops"

    # Embedding & Reranker models
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_batch_size: int = 4
    reranker_batch_size: int = 8

    # Retrieval knobs
    bm25_top_k: int = 50
    vector_top_k: int = 30
    rrf_k: int = 60
    rerank_top_k: int = 8

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    s = Settings()
    # Push model dirs to environment so HuggingFace/PaddleOCR auto-discover them
    os.environ["HF_HOME"] = s.hf_home
    os.environ["TRANSFORMERS_CACHE"] = s.transformers_cache
    os.environ["PADDLEOCR_HOME"] = s.paddleocr_home
    os.makedirs(s.hf_home, exist_ok=True)
    os.makedirs(s.paddleocr_home, exist_ok=True)
    os.makedirs(s.pages_dir, exist_ok=True)
    os.makedirs(s.crops_dir, exist_ok=True)
    return s
