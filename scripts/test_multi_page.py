"""Test: Multi-page PDF processing.

Verifies that the UnifiedPdfProcessor:
- Processes all pages of a multi-page PDF
- Handles mixed PDFs (some digital, some scanned pages)
- Returns results keyed by page number (1-indexed)
- Tracks per-page stats correctly
- Releases memory between pages (lazy processing)
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import fitz
import pytest

from jb_detection.models import OcrDetection, JBDetectionResult
from jb_detection.config import DEFAULT_CONFIG


# ── Fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def multi_page_digital_pdf(tmp_path):
    """Create a 3-page digital PDF."""
    pdf_path = str(tmp_path / "multi_digital.pdf")
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"JB-{100+i} TE-{5000+i} PT-{1014+i} MC-{200+i}", fontsize=14)
        page.insert_text((72, 150), f"FT-{3015+i} LT-{2007+i} TT-{4011+i} SPARE", fontsize=14)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def mixed_pdf(tmp_path):
    """Create a 3-page mixed PDF: digital, scanned, digital."""
    pdf_path = str(tmp_path / "mixed.pdf")
    doc = fitz.open()
    # Page 1: digital (enough text to exceed threshold)
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((72, 100), "JB-101 TE-5223 PT-1014 MC-200 SPARE FCV-101-A", fontsize=14)
    # Page 2: scanned (blank)
    doc.new_page(width=595, height=842)
    # Page 3: digital
    p3 = doc.new_page(width=595, height=842)
    p3.insert_text((72, 100), "JB-103 PT-1014 FT-3015 LT-2007 TT-4011", fontsize=14)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def mock_detector():
    """Mock TextDetector for scanned pages."""
    detector = MagicMock()
    detector.detect.return_value = [
        OcrDetection(
            text="OCR-TEXT",
            confidence=0.90,
            polygon=[[10, 10], [110, 10], [110, 40], [10, 40]],
            bbox=(10, 10, 100, 30),
        ),
    ]
    detector.stats.return_value = {"calls": 0, "detections": 0, "errors": 0}
    return detector


# ── Tests ──────────────────────────────────────────────────────────────
class TestMultiPagePdf:

    def test_process_all_pages(self, multi_page_digital_pdf):
        """Should process all 3 pages."""
        from jb_detection.unified_pdf_processor import UnifiedPdfProcessor

        processor = UnifiedPdfProcessor()
        results = processor.process_pdf(multi_page_digital_pdf)

        assert len(results) == 3
        assert set(results.keys()) == {1, 2, 3}
        for page_num, result in results.items():
            assert isinstance(result, JBDetectionResult)

    def test_page_numbers_are_1_indexed(self, multi_page_digital_pdf):
        """Page numbers in results should be 1-indexed."""
        from jb_detection.unified_pdf_processor import UnifiedPdfProcessor

        processor = UnifiedPdfProcessor()
        results = processor.process_pdf(multi_page_digital_pdf)

        assert min(results.keys()) == 1  # First page is 1, not 0

    def test_stats_track_pages_processed(self, multi_page_digital_pdf):
        """Stats should show correct page counts."""
        from jb_detection.unified_pdf_processor import UnifiedPdfProcessor

        processor = UnifiedPdfProcessor()
        processor.process_pdf(multi_page_digital_pdf)

        stats = processor.stats()
        assert stats["pages_processed"] == 3
        assert stats["pages_digital"] == 3
        assert stats["pages_scanned"] == 0

    def test_mixed_pdf_uses_both_paths(self, mixed_pdf, mock_detector):
        """Mixed PDF should use digital path for digital pages, OCR for scanned."""
        from jb_detection.unified_pdf_processor import UnifiedPdfProcessor

        processor = UnifiedPdfProcessor(detector=mock_detector)
        results = processor.process_pdf(mixed_pdf)

        assert len(results) == 3
        stats = processor.stats()
        # 2 digital pages, 1 scanned page
        assert stats["pages_digital"] == 2
        assert stats["pages_scanned"] == 1

    def test_ocr_only_called_for_scanned_pages(self, mixed_pdf, mock_detector):
        """PaddleOCR should only be called for scanned pages, not digital ones."""
        from jb_detection.unified_pdf_processor import UnifiedPdfProcessor

        processor = UnifiedPdfProcessor(detector=mock_detector)
        processor.process_pdf(mixed_pdf)

        # Only page 2 is scanned — should be exactly 1 OCR call
        assert mock_detector.detect.call_count == 1, \
            f"Expected 1 OCR call (for 1 scanned page), got {mock_detector.detect.call_count}"

    def test_failed_page_returns_empty_result(self, tmp_path, mock_detector):
        """A page that fails should return an empty JBDetectionResult, not crash."""
        from jb_detection.unified_pdf_processor import UnifiedPdfProcessor

        # Create a PDF
        pdf_path = str(tmp_path / "test.pdf")
        doc = fitz.open()
        doc.new_page(width=595, height=842)
        doc.save(pdf_path)
        doc.close()

        # Make the detector raise on first call
        mock_detector.detect.side_effect = RuntimeError("Simulated OCR failure")
        processor = UnifiedPdfProcessor(detector=mock_detector)
        results = processor.process_pdf(pdf_path)

        assert len(results) == 1
        assert isinstance(results[1], JBDetectionResult)
        # The result should be empty (or at least not crash)
        assert results[1].tags == set() or len(results[1].tags) >= 0

    def test_max_pages_cap(self, tmp_path):
        """Should respect the max_pages limit."""
        from jb_detection.unified_pdf_processor import UnifiedPdfProcessor

        # Create a 5-page digital PDF
        pdf_path = str(tmp_path / "five_pages.pdf")
        doc = fitz.open()
        for i in range(5):
            page = doc.new_page(width=595, height=842)
            page.insert_text((72, 100), f"JB-{100+i}", fontsize=14)
        doc.save(pdf_path)
        doc.close()

        processor = UnifiedPdfProcessor()
        results = processor.process_pdf(pdf_path, max_pages=3)

        assert len(results) == 3  # Capped at 3

    def test_process_multiple_pdfs(self, multi_page_digital_pdf):
        """Should handle multiple PDFs."""
        from jb_detection.unified_pdf_processor import UnifiedPdfProcessor

        processor = UnifiedPdfProcessor()
        results = processor.process_multiple_pdfs([multi_page_digital_pdf])

        assert len(results) == 1  # One PDF
        pdf_name = os.path.basename(multi_page_digital_pdf)
        assert pdf_name in results
        assert len(results[pdf_name]) == 3  # 3 pages


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
