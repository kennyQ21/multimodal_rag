"""
OCR Singleton — PaddleOCR primary, pytesseract fallback.

Rules:
- Load once per process, reuse globally (singleton).
- Never instantiate per request.
- OCR only cropped regions, not full pages.
- Cache OCR results in the DB via image hash.
"""
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# ── Singleton instances ────────────────────────────────────────────────────────
_paddle_ocr_instance = None
_tesseract_available: Optional[bool] = None


def get_ocr() -> "PaddleOCR":  # noqa: F821
    """Return the cached PaddleOCR singleton. Downloads weights once."""
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        logger.info("Initialising PaddleOCR (first call — may download weights)…")
        from paddleocr import PaddleOCR  # imported lazily to avoid slow startup
        _paddle_ocr_instance = PaddleOCR(
            use_gpu=False,
            use_angle_cls=False,   # skip angle classifier to save memory
            show_log=False,
            lang="en",
        )
        logger.info("PaddleOCR ready.")
    return _paddle_ocr_instance


def _is_tesseract_available() -> bool:
    global _tesseract_available
    if _tesseract_available is None:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            _tesseract_available = True
        except Exception:
            _tesseract_available = False
            logger.warning("pytesseract not available — fallback OCR disabled.")
    return _tesseract_available


# ── Public API ─────────────────────────────────────────────────────────────────

def run_ocr(image: np.ndarray) -> str:
    """
    Run OCR on a cropped image region.

    Flow:
        PaddleOCR
          └─ success → return text
          └─ failure → pytesseract fallback
                        └─ failure → return ""
    """
    # ── Primary: PaddleOCR ────────────────────────────────────────────────────
    try:
        ocr = get_ocr()
        result = ocr.ocr(image, cls=False)
        if result and result[0]:
            lines = [line[1][0] for line in result[0] if line and line[1]]
            return " ".join(lines).strip()
    except Exception as e:
        logger.warning(f"PaddleOCR failed: {e} — trying pytesseract fallback.")

    # ── Fallback: pytesseract ─────────────────────────────────────────────────
    if _is_tesseract_available():
        try:
            import pytesseract
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image)
            text = pytesseract.image_to_string(pil_img).strip()
            return text
        except Exception as e:
            logger.warning(f"pytesseract fallback also failed: {e}")

    return ""
