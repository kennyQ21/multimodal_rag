"""
OCR Singleton — PaddleOCR primary, pytesseract fallback.

Rules:
- Load once per process, reuse globally (singleton).
- Never instantiate per request.
- OCR only cropped regions, not full pages.
- Cache OCR results in the DB via image hash.
- Gracefully falls back to pytesseract if PaddleOCR not installed.
"""
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

_paddle_ocr_instance = None
_paddle_available: Optional[bool] = None
_tesseract_available: Optional[bool] = None


def _check_paddle() -> bool:
    global _paddle_available
    if _paddle_available is None:
        try:
            import paddleocr  # noqa: F401
            import paddlepaddle  # noqa: F401
            _paddle_available = True
        except ImportError:
            _paddle_available = False
            logger.warning("PaddleOCR not installed — will use pytesseract fallback.")
    return _paddle_available


def get_ocr():
    """Return the cached PaddleOCR singleton (downloads weights on first call)."""
    global _paddle_ocr_instance
    if not _check_paddle():
        return None
    if _paddle_ocr_instance is None:
        logger.info("Initialising PaddleOCR (first call — may download weights)…")
        from paddleocr import PaddleOCR
        _paddle_ocr_instance = PaddleOCR(
            use_gpu=False,
            use_angle_cls=False,
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
            
            tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            import os
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path

            pytesseract.get_tesseract_version()
            _tesseract_available = True
        except Exception:
            _tesseract_available = False
            logger.warning("pytesseract not available — OCR will return empty strings.")
    return _tesseract_available


def run_ocr(image: np.ndarray) -> str:
    """
    Run OCR on a cropped image region.
    Flow: PaddleOCR → pytesseract fallback → "" if both fail.
    """
    # ── Primary: PaddleOCR ────────────────────────────────────────────────────
    if _check_paddle():
        try:
            ocr = get_ocr()
            if ocr:
                result = ocr.ocr(image, cls=False)
                if result and result[0]:
                    lines = [line[1][0] for line in result[0] if line and line[1]]
                    text = " ".join(lines).strip()
                    if text:
                        return text
        except Exception as e:
            logger.warning(f"PaddleOCR failed: {e} — trying pytesseract.")

    # ── Fallback: pytesseract ─────────────────────────────────────────────────
    if _is_tesseract_available():
        try:
            import pytesseract
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image)
            text = pytesseract.image_to_string(pil_img, config="--psm 6").strip()
            return text
        except Exception as e:
            logger.warning(f"pytesseract also failed: {e}")

    return ""
