"""
Setup script — downloads and caches PaddleOCR and BGE models on first run.
Run once after pip install: python setup_models.py
"""
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

from app.config import get_settings

settings = get_settings()  # also sets env vars + creates dirs

def download_ocr():
    log.info("=== Initialising PaddleOCR (will download weights if not cached) ===")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_gpu=False, use_angle_cls=False, show_log=False, lang="en")
    log.info("PaddleOCR weights cached.")

def download_embedder():
    log.info("=== Downloading BGE embedding model ===")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        settings.embedding_model,
        device="cpu",
        cache_folder=settings.hf_home,
    )
    _ = model.encode("warmup", normalize_embeddings=True)
    log.info("Embedding model cached.")

def download_reranker():
    log.info("=== Downloading BGE reranker model ===")
    from FlagEmbedding import FlagReranker
    reranker = FlagReranker(settings.reranker_model, use_fp16=False)
    _ = reranker.compute_score([["test query", "test passage"]])
    log.info("Reranker model cached.")

if __name__ == "__main__":
    download_ocr()
    download_embedder()
    download_reranker()
    log.info("=== All models cached. Run 'python run.py' to start the server. ===")
