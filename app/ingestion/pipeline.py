"""
PDF Ingestion Pipeline

Flow per page:
    PyMuPDF → render page image → LayoutParser → crop regions
    → OCR each crop (PaddleOCR + fallback) → OCR cache check
    → semantic chunking → embed → store in PostgreSQL
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

# ── LayoutParser model singleton ───────────────────────────────────────────────
_lp_model = None

def get_layout_model():
    global _lp_model
    if _lp_model is None:
        logger.info("Loading LayoutParser model…")
        import layoutparser as lp
        _lp_model = lp.PaddleDetectionLayoutModel(
            config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
            threshold=0.45,
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            enforce_cpu=True,
        )
        logger.info("LayoutParser ready.")
    return _lp_model


# ── Helpers ────────────────────────────────────────────────────────────────────

def _hash_image(img: np.ndarray) -> str:
    return hashlib.md5(img.tobytes()).hexdigest()


def _get_cached_ocr(img_hash: str, db) -> str | None:
    row = db.query(OCRCache).filter(OCRCache.image_hash == img_hash).first()
    return row.ocr_text if row else None


def _cache_ocr(img_hash: str, text: str, db):
    db.merge(OCRCache(image_hash=img_hash, ocr_text=text, created_at=datetime.utcnow()))
    db.commit()


def _figure_label(figure_counter: int) -> str:
    return f"Fig {figure_counter}"


# ── Chunk flush ────────────────────────────────────────────────────────────────

def _flush_chunk(doc_id: str, proc_id: str, proc_title: str,
                 page_start: int, page_end: int,
                 steps: list[dict], db):
    """Embed steps retrieval text and persist a Chunk row."""
    if not steps:
        return

    from app.retrieval.embeddings import embed_texts
    settings = get_settings()

    # Build retrieval text: instruction + OCR captions for images
    parts = []
    for s in steps:
        if s.get("instruction"):
            parts.append(s["instruction"])
        for img_id in s.get("images", []):
            img_row = db.query(Image).filter(Image.id == img_id).first()
            if img_row and img_row.caption:
                parts.append(img_row.caption)

    retrieval_text = " ".join(parts).strip()
    if not retrieval_text:
        return

    vectors = embed_texts([retrieval_text], batch_size=settings.embedding_batch_size)
    embedding = vectors[0]

    chunk = Chunk(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        procedure_id=proc_id,
        procedure_title=proc_title,
        page_start=page_start,
        page_end=page_end,
        steps=steps,
        retrieval_text=retrieval_text,
        embedding=embedding,
    )
    db.add(chunk)
    db.commit()
    logger.debug(f"Flushed chunk for procedure '{proc_title}' pages {page_start}-{page_end}.")


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
    logger.info(f"Ingesting '{filename}' ({total_pages} pages) as doc_id={doc_id}")

    lp_model = get_layout_model()

    # State for semantic chunking
    current_proc_id = str(uuid.uuid4())
    current_proc_title = "Introduction"
    current_proc_page_start = 1
    current_steps: list[dict] = []
    current_step_num = 1
    figure_counter = 1

    for page_idx in range(total_pages):
        page = pdf[page_idx]
        page_num = page_idx + 1

        # ── Skip truly blank pages ─────────────────────────────────────────────
        if not page.get_text().strip() and not page.get_images(full=True):
            logger.debug(f"Page {page_num}: blank — skipping.")
            continue

        # ── Render page to image ───────────────────────────────────────────────
        pix = page.get_pixmap(dpi=150)
        page_img_path = os.path.join(settings.pages_dir, f"page_{doc_id}_{page_num}.png")
        pix.save(page_img_path)

        page_row = Page(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            page_number=page_num,
            image_path=page_img_path,
        )
        db.add(page_row)
        db.commit()

        # ── Layout detection ───────────────────────────────────────────────────
        img_cv = cv2.imread(page_img_path)
        if img_cv is None:
            logger.warning(f"Could not read rendered image for page {page_num}.")
            continue

        layout = lp_model.detect(img_cv)
        # Sort top-to-bottom for reading order
        blocks = sorted(layout, key=lambda b: b.coordinates[1])

        for block in blocks:
            x1, y1, x2, y2 = [int(v) for v in block.coordinates]
            block_type = block.type

            crop = img_cv[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            img_hash = _hash_image(crop)

            # ── OCR with cache ─────────────────────────────────────────────────
            ocr_text = _get_cached_ocr(img_hash, db)
            if ocr_text is None:
                ocr_text = run_ocr(crop)
                _cache_ocr(img_hash, ocr_text, db)

            # ── Handle each block type ─────────────────────────────────────────
            if block_type == "Title":
                # Flush current procedure chunk when a new title appears
                if current_steps:
                    _flush_chunk(
                        doc_id, current_proc_id, current_proc_title,
                        current_proc_page_start, page_num, current_steps, db
                    )
                    current_steps = []
                    current_step_num = 1

                current_proc_id = str(uuid.uuid4())
                current_proc_title = ocr_text or f"Procedure (page {page_num})"
                current_proc_page_start = page_num
                logger.debug(f"New procedure: '{current_proc_title}'")

            elif block_type in ("Text", "List"):
                if len(ocr_text) > 8:
                    current_steps.append({
                        "step": current_step_num,
                        "instruction": ocr_text,
                        "images": [],
                    })
                    current_step_num += 1

            elif block_type in ("Figure", "Table"):
                # Save crop to disk
                crop_path = os.path.join(settings.crops_dir, f"{img_hash}.png")
                if not os.path.exists(crop_path):
                    cv2.imwrite(crop_path, crop)

                label = _figure_label(figure_counter)
                figure_counter += 1

                img_row = Image(
                    id=img_hash,
                    page_id=page_row.id,
                    page_number=page_num,
                    bbox=[x1, y1, x2, y2],
                    path=crop_path,
                    figure_label=label,
                    caption=ocr_text,
                )
                db.add(img_row)
                db.commit()

                # Attach image to last instruction step (or create orphan step)
                if current_steps:
                    current_steps[-1]["images"].append(img_hash)
                else:
                    current_steps.append({
                        "step": current_step_num,
                        "instruction": "",
                        "images": [img_hash],
                    })
                    current_step_num += 1

        logger.info(f"Page {page_num}/{total_pages} processed.")

    # Flush final procedure chunk
    if current_steps:
        _flush_chunk(
            doc_id, current_proc_id, current_proc_title,
            current_proc_page_start, total_pages, current_steps, db
        )

    pdf.close()
    db.close()
    logger.info(f"Ingestion complete for doc_id={doc_id}.")
    return doc_id
