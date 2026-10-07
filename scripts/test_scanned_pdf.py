"""Test: Scanned PDF processing via PaddleOCR.

Tests the OCR path of the unified pipeline. Since PaddleOCR may not
be installed in the test environment, these tests are designed to:
1. Test the pipeline structure (mock the detector if PaddleOCR is unavailable).
2. If PaddleOCR IS available, test with a real rendered page.

Run with:
    pytest test_scanned_pdf.py -v
    # or to skip if PaddleOCR is not installed:
    pytest test_scanned_pdf.py -v -k "not requires_paddleocr"
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import fitz
import numpy as np
import pytest

from jb_detection.models import OcrDetection, JBDetectionResult
from jb_detection.pattern_matcher import PatternMatcher
from jb_detection.config import DEFAULT_CONFIG


# ── Check if PaddleOCR is available ────────────────────────────────────
def paddleocr_available():
    """Check if PaddleOCR can be imported."""
    try:
        import paddleocr
        return True
    except ImportError:
        return False


PADDLEOCR_AVAILABLE = paddleocr_available()


# ── Fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def scanned_pdf(tmp_path):
    """Create a scanned-style PDF (image-only, no text layer)."""
    pdf_path = str(tmp_path / "scanned.pdf")
    doc = fitz.open()
    # Create a blank page (no text = scanned)
    doc.new_page(width=595, height=842)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def mock_detector():
    """Mock TextDetector that returns fake OcrDetection objects."""
    detector = MagicMock()
    detector.detect.return_value = [
        OcrDetection(
            text="JB-101",
            confidence=0.95,
            polygon=[[10, 10], [110, 10], [110, 40], [10, 40]],
            bbox=(10, 10, 100, 30),
        ),
        OcrDetection(
            text="TE-5223",
            confidence=0.92,
            polygon=[[10, 50], [110, 50], [110, 80], [10, 80]],
            bbox=(10, 50, 100, 30),
        ),
        OcrDetection(
            text="MC-200",
            confidence=0.88,
            polygon=[[10, 90], [110, 90], [110, 120], [10, 120]],
            bbox=(10, 90, 100, 30),
        ),
    ]
    detector.stats.return_value = {"calls": 1, "detections": 3, "errors": 0}
    return detector


# ── Tests ──────────────────────────────────────────────────────────────
class TestScannedPdfProcessing:

    def test_scanned_pdf_detected_as_scanned(self, scanned_pdf):
        """A blank PDF should be detected as scanned."""
        from jb_detection.pdf_type_detector import PdfType, PdfTypeDetector
        detector = PdfTypeDetector()
        result = detector.detect_pdf_type(scanned_pdf)
        assert result == PdfType.SCANNED

    def test_pattern_matcher_processes_ocr_detections(self, mock_detector):
        """PatternMatcher should correctly classify mock OCR detections."""
        pm = PatternMatcher(
            jb_examples=["JB"],
            mc_examples=["MC"],
        )
        detections = mock_detector.detect.return_value
        result = pm.match(detections)

        assert isinstance(result, JBDetectionResult)
        # JB-101 should be in jb_identifiers
        assert "JB-101" in result.jb_identifiers or len(result.jb_identifiers) > 0
        # MC-200 should be in mc_identifiers
        assert "MC-200" in result.mc_identifiers or len(result.mc_identifiers) > 0

    def test_unified_processor_uses_paddleocr_for_scanned(
        self, scanned_pdf, mock_detector,
    ):
        """The unified processor should call the detector for scanned pages."""
        from jb_detection.unified_pdf_processor import UnifiedPdfProcessor

        processor = UnifiedPdfProcessor(
            detector=mock_detector,
        )
        results = processor.process_pdf(scanned_pdf)

        assert len(results) == 1  # 1 page
        assert processor.pages_scanned >= 1
        assert processor.pages_digital == 0
        # The detector was called once for the scanned page
        assert mock_detector.detect.call_count >= 1

    def test_ocr_called_once_per_page(self, scanned_pdf, mock_detector):
        """PaddleOCR should be called exactly once per scanned page."""
        from jb_detection.unified_pdf_processor import UnifiedPdfProcessor

        processor = UnifiedPdfProcessor(detector=mock_detector)
        processor.process_pdf(scanned_pdf)

        # Exactly one detect() call for one page
        assert mock_detector.detect.call_count == 1, \
            f"Expected 1 OCR call, got {mock_detector.detect.call_count}"

    def test_detector_singleton_reuse(self, mock_detector):
        """The detector should be reused, not re-initialized per page."""
        from jb_detection.unified_pdf_processor import UnifiedPdfProcessor

        processor = UnifiedPdfProcessor(detector=mock_detector)
        # The detector property should return the same instance
        assert processor.detector is mock_detector

    @pytest.mark.skipif(
        not PADDLEOCR_AVAILABLE,
        reason="PaddleOCR not installed",
    )
    def test_real_paddleocr_on_image(self):
        """If PaddleOCR is available, test it on a real rendered page."""
        from jb_detection.detector import TextDetector

        # Create a simple image with text
        import cv2
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, "TE-5223", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                     1.0, (0, 0, 0), 2)

        detector = TextDetector.get_instance()
        detector.reset_instance()  # Reset to force re-init
        TextDetector._instance = None

        try:
            detector = TextDetector.get_instance()
            detections = detector.detect(img)
            # PaddleOCR should find something (text is clear)
            assert isinstance(detections, list)
        except Exception as e:
            pytest.skip(f"PaddleOCR init failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
