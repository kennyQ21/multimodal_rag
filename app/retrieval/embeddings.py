"""
Embedding Model Singleton — BAAI/bge-large-en-v1.5 (CPU, 1024-dim).

Rules:
- Single instance per process.
- Loaded lazily on first encode() call.
- Weights cached in models/huggingface/.
"""
import logging
from typing import List

logger = logging.getLogger(__name__)

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        _embedding_model = DefaultEmbeddingFunction()
        logger.info(f"Loaded ONNX DefaultEmbeddingFunction")
    return _embedding_model


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    if not texts:
        return []
    model = get_embedding_model()
    # DefaultEmbeddingFunction expects List[str] and returns List[List[float]]
    return model(texts)
