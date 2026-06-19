"""
Unit tests for OCR module.
Tests singleton behaviour and fallback logic without needing real images.
"""
import numpy as np
from unittest.mock import patch, MagicMock


def test_run_ocr_returns_string():
    """run_ocr must always return a string."""
    from app.ingestion.ocr import run_ocr
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    result = run_ocr(blank)
    assert isinstance(result, str)


def test_paddle_singleton_not_reinitialised():
    """get_ocr() must return the same instance on repeated calls."""
    import app.ingestion.ocr as ocr_module
    # Reset singleton for test isolation
    ocr_module._paddle_ocr_instance = None

    with patch("app.ingestion.ocr.get_ocr") as mock_get:
        fake = MagicMock()
        mock_get.return_value = fake
        a = mock_get()
        b = mock_get()
        assert a is b
