"""
Embedding Model Singleton — BAAI/bge-large-en-v1.5 (CPU, 1024-dim).

Rules:
- Single instance per process.
- Loaded lazily on first encode() call.
- Weights cached in models/huggingface/.
"""
import logging
from app.config import get_settings

logger = logging.getLogger(__name__)

_embedder = None


def get_embedding_model():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        settings = get_settings()
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _embedder = SentenceTransformer(
            settings.embedding_model,
            device="cpu",
            cache_folder=settings.hf_home,
        )
        logger.info("Embedding model ready.")
    return _embedder


def embed_texts(texts: list[str], batch_size: int = 4) -> list[list[float]]:
    """Embed a list of texts and return as list of float vectors."""
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()
