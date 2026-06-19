"""
PDF Ingestion Pipeline

Uses PyMuPDF for text block + image extraction.
No external layout model required — uses PyMuPDF's native block detection
which is reliable for structured technical PDFs.

Flow per page:
  PyMuPDF → text blocks + images → OCR image crops → semantic chunks → embed → store
"""
import os
import uuid
import hashlib
import logging
from datetime import datetime
from pathlib import Path

import fitz          # PyMuPDF
import cv2
import numpy as np

from app.config import get_settings
from app.ingestion.ocr import run_ocr
from app.storage.database import SessionLocal
from app.storage.models import Document, Page, Image, Chunk, OCRCache

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _hash_image(img: np.ndarray) -> str:
    return hashlib.md5(img.tobytes()).hexdigest()


def _get_cached_ocr(img_hash: str, db) -> str | None:
    row = db.query(OCRCache).filter(OCRCache.image_hash == img_hash).first()
    return row.ocr_text if row else None


def _cache_ocr(img_hash: str, text: str, db):
    db.merge(OCRCache(image_hash=img_hash, ocr_text=text, created_at=datetime.utcnow()))
    db.commit()


# ── Chunk flush ────────────────────────────────────────────────────────────────

def _flush_chunk(doc_id, proc_id, proc_title, page_start, page_end, steps, db):
    if not steps:
        return
    # Filter out empty steps
    valid_steps = [s for s in steps if s.get("instruction") or s.get("images")]
    if not valid_steps:
        return

    from app.retrieval.embeddings import embed_texts
    settings = get_settings()

    parts = []
    for s in valid_steps:
        if s.get("instruction"):
            parts.append(s["instruction"])
        for img_id in s.get("images", []):
            cached = _get_cached_ocr(img_id, db)
            if cached:
                parts.append(cached)

    retrieval_text = " ".join(parts).strip()
    if not retrieval_text:
        retrieval_text = proc_title or "procedure"

    chunk_id = str(uuid.uuid4())
    chunk = Chunk(
        id=chunk_id,
        document_id=doc_id,
        procedure_id=proc_id,
        procedure_title=proc_title,
        page_start=page_start,
        page_end=page_end,
        steps=valid_steps,
        retrieval_text=retrieval_text,
    )
    db.add(chunk)
    db.commit()
    
    logger.debug(f"Flushed chunk '{proc_title}' pages {page_start}-{page_end} with {len(valid_steps)} steps.")


# ── Main ingestion ─────────────────────────────────────────────────────────────

def process_pdf(file_path: str) -> str:
    settings = get_settings()
    db = SessionLocal()

    doc_id = str(uuid.uuid4())
    filename = Path(file_path).name
    pdf = fitz.open(file_path)
    total_pages = len(pdf)

    doc_row = Document(
        id=doc_id,
        filename=filename,
        total_pages=total_pages,
        uploaded_at=datetime.utcnow(),
    )
    db.add(doc_row)
    db.commit()
    logger.info(f"Ingesting '{filename}' ({total_pages} pages) → doc_id={doc_id}")

    # Chunking state
    current_proc_id = str(uuid.uuid4())
    current_proc_title = "Introduction"
    current_proc_page_start = 1
    current_steps: list[dict] = []
    current_step_num = 1
    figure_counter = 1

    for page_idx in range(total_pages):
        page = pdf[page_idx]
        page_num = page_idx + 1

        # Render page image (for cropping)
        pix = page.get_pixmap(dpi=150)
        page_img_path = os.path.join(settings.pages_dir, f"page_{doc_id}_{page_num}.png")
        pix.save(page_img_path)
        img_cv = cv2.imread(page_img_path)

        # Store page row
        page_row = Page(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            page_number=page_num,
            image_path=page_img_path,
        )
        db.add(page_row)
        db.commit()

        # OCR the full page
        ocr_text = run_ocr(img_cv)
        
        # Save crop of the whole page as a fallback image if no other images exist
        img_hash = _hash_image(img_cv)
        crop_path = os.path.join(settings.crops_dir, f"{img_hash}.png")
        if not os.path.exists(crop_path):
            cv2.imwrite(crop_path, img_cv)

        img_row = Image(
            id=img_hash,
            page_id=page_row.id,
            page_number=page_num,
            bbox=[0, 0, img_cv.shape[1], img_cv.shape[0]],
            path=crop_path,
            figure_label=f"Page {page_num}",
            caption=ocr_text[:200],
        )
        try:
            db.merge(img_row)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to merge image {img_hash}: {e}")

        # Treat each page as a separate procedure/chunk
        step = {
            "step": 1,
            "instruction": ocr_text,
            "images": [img_hash]
        }
        
        proc_id = str(uuid.uuid4())
        proc_title = f"Page {page_num} Procedures"
        
        _flush_chunk(doc_id, proc_id, proc_title, page_num, page_num, [step], db)
        logger.info(f"Page {page_num}/{total_pages} done.")



    pdf.close()
    db.close()
    logger.info(f"Ingestion complete → doc_id={doc_id}")
    return doc_id
