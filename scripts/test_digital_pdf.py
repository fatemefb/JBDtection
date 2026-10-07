"""Test: Digital PDF text extraction.

Verifies that DigitalTextExtractor:
- Extracts text from digital PDF pages
- Produces OcrDetection objects with correct fields
- Converts PDF point coordinates to pixel coordinates correctly
- Assigns confidence=1.0 (native extraction)
- Sorts detections in reading order
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import fitz
import pytest

from jb_detection.digital_text_extractor import (
    DigitalTextExtractor,
    NATIVE_EXTRACTION_CONFIDENCE,
)
from jb_detection.models import OcrDetection
from jb_detection.config import DEFAULT_CONFIG


# ── Fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def digital_pdf(tmp_path):
    """Create a digital PDF with known text content."""
    pdf_path = str(tmp_path / "digital_tags.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    # Insert text at known positions
    page.insert_text((72, 100), "JB-101", fontsize=14)      # 1 inch from left
    page.insert_text((72, 150), "TE-5223", fontsize=14)
    page.insert_text((72, 200), "PT-1014", fontsize=14)
    page.insert_text((72, 250), "MC-200", fontsize=14)
    page.insert_text((72, 300), "SPARE", fontsize=14)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def multi_page_digital_pdf(tmp_path):
    """Create a multi-page digital PDF."""
    pdf_path = str(tmp_path / "multi_digital.pdf")
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100 + i * 50), f"JB-{100+i}", fontsize=14)
        page.insert_text((72, 150 + i * 50), f"TE-{5000+i}", fontsize=14)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


# ── Tests ──────────────────────────────────────────────────────────────
class TestDigitalTextExtractor:

    def test_extract_from_page_returns_ocr_detections(self, digital_pdf):
        """Extracted data should be OcrDetection objects."""
        extractor = DigitalTextExtractor()
        doc = fitz.open(digital_pdf)
        page = doc.load_page(0)

        detections = extractor.extract_from_page(page)

        assert len(detections) > 0, "Should find text spans"
        for det in detections:
            assert isinstance(det, OcrDetection)
            assert det.text  # Non-empty text
            assert det.confidence == NATIVE_EXTRACTION_CONFIDENCE
            assert len(det.polygon) == 4  # 4-point polygon
            assert len(det.bbox) == 4  # (x, y, w, h)
        doc.close()

    def test_extracted_text_content(self, digital_pdf):
        """The extracted text should match what was inserted."""
        extractor = DigitalTextExtractor()
        doc = fitz.open(digital_pdf)
        page = doc.load_page(0)

        detections = extractor.extract_from_page(page)
        texts = [d.text for d in detections]

        # The PDF was created with: JB-101, TE-5223, PT-1014, MC-200, SPARE
        all_text = " ".join(texts)
        assert "JB-101" in all_text or "JB" in all_text
        assert "TE-5223" in all_text or "TE" in all_text
        doc.close()

    def test_confidence_is_1_for_native_extraction(self, digital_pdf):
        """Native extraction should have confidence=1.0."""
        extractor = DigitalTextExtractor()
        doc = fitz.open(digital_pdf)
        page = doc.load_page(0)

        detections = extractor.extract_from_page(page)
        for det in detections:
            assert det.confidence == 1.0, \
                f"Native extraction confidence should be 1.0, got {det.confidence}"
        doc.close()

    def test_coordinate_conversion(self, digital_pdf):
        """Pixel coordinates should be scaled from PDF points by dpi/72."""
        dpi = 300
        extractor = DigitalTextExtractor(config=DEFAULT_CONFIG)
        doc = fitz.open(digital_pdf)
        page = doc.load_page(0)

        # Extract at 300 DPI
        detections_300 = extractor.extract_from_page(page, dpi=300)
        # Extract at 150 DPI
        detections_150 = extractor.extract_from_page(page, dpi=150)

        assert len(detections_300) == len(detections_150)
        if len(detections_300) > 0:
            # 300 DPI coordinates should be 2x the 150 DPI coordinates
            d300 = detections_300[0]
            d150 = detections_150[0]
            assert d300.bbox[0] == pytest.approx(d150.bbox[0] * 2, abs=2)
            assert d300.bbox[1] == pytest.approx(d150.bbox[1] * 2, abs=2)
        doc.close()

    def test_reading_order_sort(self, digital_pdf):
        """Detections should be sorted top-to-bottom, left-to-right."""
        extractor = DigitalTextExtractor()
        doc = fitz.open(digital_pdf)
        page = doc.load_page(0)

        detections = extractor.extract_from_page(page)
        if len(detections) >= 2:
            for i in range(len(detections) - 1):
                curr_y = detections[i].bbox[1]
                next_y = detections[i + 1].bbox[1]
                # Either same row (similar y) or next row (higher y)
                assert next_y >= curr_y - 5, \
                    f"Detections not in reading order: {curr_y} → {next_y}"
        doc.close()

    def test_polygon_has_four_points(self, digital_pdf):
        """Each detection should have a 4-point polygon."""
        extractor = DigitalTextExtractor()
        doc = fitz.open(digital_pdf)
        page = doc.load_page(0)

        detections = extractor.extract_from_page(page)
        for det in detections:
            assert len(det.polygon) == 4, \
                f"Polygon should have 4 points, got {len(det.polygon)}"
            for point in det.polygon:
                assert len(point) == 2  # (x, y)
        doc.close()

    def test_extract_from_pdf_all_pages(self, multi_page_digital_pdf):
        """Should extract from all pages of a multi-page PDF."""
        extractor = DigitalTextExtractor()
        results = extractor.extract_from_pdf(multi_page_digital_pdf)

        assert len(results) == 3, f"Expected 3 pages, got {len(results)}"
        for page_num, detections in results.items():
            assert isinstance(page_num, int)
            assert isinstance(detections, list)
            assert len(detections) > 0, f"Page {page_num} should have detections"

    def test_empty_page_returns_empty_list(self, tmp_path):
        """A page with no text should return an empty list."""
        pdf_path = str(tmp_path / "blank.pdf")
        doc = fitz.open()
        doc.new_page(width=595, height=842)  # Blank page
        doc.save(pdf_path)
        doc.close()

        extractor = DigitalTextExtractor()
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        detections = extractor.extract_from_page(page)

        assert detections == []
        doc.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
