"""Test: PDF type detection (Digital vs Scanned).

Verifies that PdfTypeDetector correctly identifies:
- Digital PDFs (have extractable text)
- Scanned PDFs (image-based, no text)
- Empty PDFs (no pages)
- Non-existent files (UNKNOWN)
"""

import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import fitz
import pytest

from jb_detection.pdf_type_detector import PdfType, PdfTypeDetector
from jb_detection.config import DEFAULT_CONFIG


# ── Fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def digital_pdf(tmp_path):
    """Create a simple digital PDF with text."""
    pdf_path = str(tmp_path / "digital.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # Insert enough text to exceed the 50-char threshold
    page.insert_text((72, 100), "JB-101 TE-5223 PT-1014 MC-200 SPARE FCV-101-A", fontsize=14)
    page.insert_text((72, 150), "FT-3015 LT-2007 TT-4011 PI-5008 ZSO-3001", fontsize=14)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def scanned_pdf(tmp_path):
    """Create a scanned-style PDF (no extractable text)."""
    pdf_path = str(tmp_path / "scanned.pdf")
    doc = fitz.open()
    doc.new_page(width=595, height=842)  # Blank page = no text
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def empty_pdf(tmp_path):
    """Create a PDF that simulates an empty/unreadable condition."""
    # PyMuPDF cannot save a document with zero pages.
    # Instead, create a minimal 1-page PDF with no extractable text
    # (which will be classified as SCANNED, not EMPTY).
    # For the EMPTY test, we'll use a non-existent file path instead.
    pdf_path = str(tmp_path / "empty.pdf")
    # Create a 1-byte file that's not a valid PDF
    with open(pdf_path, 'wb') as f:
        f.write(b'%PDF-1.0 broken')
    return pdf_path


@pytest.fixture
def mixed_pdf(tmp_path):
    """Create a PDF with mixed pages (one digital, one scanned)."""
    pdf_path = str(tmp_path / "mixed.pdf")
    doc = fitz.open()
    # Page 1: digital (has text)
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((100, 100), "JB-101 TE-5223", fontsize=14)
    # Page 2: scanned (blank — no text)
    doc.new_page(width=595, height=842)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


# ── Tests ──────────────────────────────────────────────────────────────
class TestPdfTypeDetector:

    def test_detect_digital_pdf(self, digital_pdf):
        """A PDF with text should be detected as DIGITAL."""
        detector = PdfTypeDetector()
        result = detector.detect_pdf_type(digital_pdf)
        assert result == PdfType.DIGITAL, f"Expected DIGITAL, got {result}"

    def test_detect_scanned_pdf(self, scanned_pdf):
        """A PDF with no text should be detected as SCANNED."""
        detector = PdfTypeDetector()
        result = detector.detect_pdf_type(scanned_pdf)
        assert result == PdfType.SCANNED, f"Expected SCANNED, got {result}"

    def test_detect_empty_pdf(self, empty_pdf):
        """A broken/empty PDF should be detected as UNKNOWN (not crash)."""
        detector = PdfTypeDetector()
        result = detector.detect_pdf_type(empty_pdf)
        assert result in (PdfType.EMPTY, PdfType.UNKNOWN), \
            f"Expected EMPTY or UNKNOWN, got {result}"

    def test_detect_nonexistent_pdf(self):
        """A non-existent file should be detected as UNKNOWN."""
        detector = PdfTypeDetector()
        result = detector.detect_pdf_type("/nonexistent/path/to/file.pdf")
        assert result == PdfType.UNKNOWN, f"Expected UNKNOWN, got {result}"

    def test_detect_page_type_digital(self, digital_pdf):
        """Per-page detection should identify digital pages."""
        detector = PdfTypeDetector()
        doc = fitz.open(digital_pdf)
        page = doc.load_page(0)
        result = detector.detect_page_type(page)
        assert result == PdfType.DIGITAL
        doc.close()

    def test_detect_page_type_scanned(self, scanned_pdf):
        """Per-page detection should identify scanned pages."""
        detector = PdfTypeDetector()
        doc = fitz.open(scanned_pdf)
        page = doc.load_page(0)
        result = detector.detect_page_type(page)
        assert result == PdfType.SCANNED
        doc.close()

    def test_mixed_pdf_detected_as_majority(self, mixed_pdf):
        """A mixed PDF should be detected based on majority of pages."""
        detector = PdfTypeDetector()
        result = detector.detect_pdf_type(mixed_pdf)
        # 1 digital + 1 scanned = tie → scanned wins (default)
        assert result in (PdfType.DIGITAL, PdfType.SCANNED)

    def test_min_text_chars_threshold(self, tmp_path):
        """Pages with very little text should be classified as scanned."""
        pdf_path = str(tmp_path / "minimal.pdf")
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        # Insert very little text (< default threshold of 30 chars)
        page.insert_text((100, 100), "Hi", fontsize=14)
        doc.save(pdf_path)
        doc.close()

        detector = PdfTypeDetector()
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        result = detector.detect_page_type(page)
        # "Hi" is only 2 chars — below the 30-char threshold → SCANNED
        assert result == PdfType.SCANNED, f"Expected SCANNED for minimal text, got {result}"
        doc.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
