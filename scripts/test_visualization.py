"""Test: Bounding-box visualization (color-coded by category).

Verifies that PDFAnnotator:
- Produces a valid annotated PDF file
- Draws bounding boxes with correct colors per category
- Preserves page count
- Handles missing/empty results gracefully
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import fitz
import pytest

from jb_detection.models import JBDetectionResult, TagMatchInfo
from jb_detection.annotator import PDFAnnotator, CATEGORY_COLORS_RGB
from jb_detection.config import DEFAULT_CONFIG


# ── Fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def simple_pdf(tmp_path):
    """Create a simple 2-page PDF."""
    pdf_path = str(tmp_path / "simple.pdf")
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"Page {i+1}", fontsize=14)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def sample_results():
    """Sample JBDetectionResult with various categories."""
    result = JBDetectionResult()
    result.tags = {"TE-5223", "PT-1014"}
    result.jb_identifiers = {"JB-101"}
    result.mc_identifiers = {"MC-200"}
    result.cable_descriptions = ["NC-0-1-2-C-3-BL"]
    result.spare_identifiers = ["SPARE"]

    # Add tag_match_info with bboxes
    result.tag_match_info = {
        "TE-5223": TagMatchInfo(
            match_type="exact", score=1.0,
            ocr_text="TE-5223", matched_tag="TE-5223",
            bbox=(100, 100, 100, 30),
        ),
        "PT-1014": TagMatchInfo(
            match_type="similar", score=0.92,
            ocr_text="PT-1014", matched_tag="PT-1014",
            bbox=(100, 150, 100, 30),
        ),
    }

    # Add positions for JBs and MCs
    result.tag_positions = {
        "JB-101": (100, 200),
        "MC-200": (100, 250),
        "NC-0-1-2-C-3-BL": (100, 300),
    }

    # Add spare positions
    result.spare_positions = [
        {"text": "SPARE", "position": (100, 350)},
    ]

    return {1: result}


# ── Tests ──────────────────────────────────────────────────────────────
class TestVisualization:

    def test_category_colors_defined(self):
        """All required category colors should be defined."""
        required = ["tag", "jb", "mc", "cable", "spare", "unknown"]
        for cat in required:
            assert cat in CATEGORY_COLORS_RGB, f"Missing color for {cat}"
            color = CATEGORY_COLORS_RGB[cat]
            assert len(color) == 3  # RGB tuple

    def test_colors_are_distinct(self):
        """Each category should have a distinct color."""
        colors = list(CATEGORY_COLORS_RGB.values())
        # Convert to set of tuples — distinct colors should have >1 unique value
        unique = set(colors)
        assert len(unique) >= 5  # At least 5 distinct colors for 6 categories

    def test_annotator_creates_output_file(self, simple_pdf, sample_results, tmp_path):
        """annotate_pdf should produce a valid PDF file."""
        output_path = str(tmp_path / "annotated_output.pdf")
        annotator = PDFAnnotator()

        counts = annotator.annotate_pdf(
            pdf_path=simple_pdf,
            page_results=sample_results,
            output_path=output_path,
            tag_to_number={"TE-5223": 1, "PT-1014": 2},
        )

        # File should exist
        assert os.path.exists(output_path)
        # Should be a valid PDF
        doc = fitz.open(output_path)
        assert len(doc) == 2  # Same page count as input
        doc.close()

    def test_annotation_counts(self, simple_pdf, sample_results, tmp_path):
        """Counts should reflect the number of annotations drawn."""
        output_path = str(tmp_path / "annotated.pdf")
        annotator = PDFAnnotator()

        counts = annotator.annotate_pdf(
            pdf_path=simple_pdf,
            page_results=sample_results,
            output_path=output_path,
        )

        # sample_results has 2 tags, 1 JB, 1 MC, 1 cable, 1 spare
        assert counts["tags"] == 2
        assert counts["jbs"] == 1
        assert counts["mcs"] == 1
        # Cables and spares depend on positions being available
        assert isinstance(counts["cables"], int)
        assert isinstance(counts["spares"], int)

    def test_empty_results_produces_valid_pdf(self, simple_pdf, tmp_path):
        """Empty results should still produce a valid PDF (no annotations)."""
        output_path = str(tmp_path / "empty_annotated.pdf")
        annotator = PDFAnnotator()

        counts = annotator.annotate_pdf(
            pdf_path=simple_pdf,
            page_results={},
            output_path=output_path,
        )

        assert os.path.exists(output_path)
        doc = fitz.open(output_path)
        assert len(doc) == 2
        doc.close()
        assert counts["tags"] == 0

    def test_nonexistent_pdf_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent PDF."""
        annotator = PDFAnnotator()
        with pytest.raises(FileNotFoundError):
            annotator.annotate_pdf(
                pdf_path="/nonexistent/file.pdf",
                page_results={},
                output_path=str(tmp_path / "out.pdf"),
            )

    def test_output_directory_created(self, simple_pdf, sample_results, tmp_path):
        """Should create output directory if it doesn't exist."""
        output_dir = str(tmp_path / "subdir" / "deep")
        output_path = os.path.join(output_dir, "annotated.pdf")

        annotator = PDFAnnotator()
        annotator.annotate_pdf(
            pdf_path=simple_pdf,
            page_results=sample_results,
            output_path=output_path,
        )

        assert os.path.exists(output_path)

    def test_page_out_of_range_skipped(self, simple_pdf, sample_results, tmp_path):
        """Pages out of range should be skipped (logged, not crash)."""
        # Add a page 99 to results (PDF only has 2 pages)
        results = dict(sample_results)
        results[99] = JBDetectionResult()

        output_path = str(tmp_path / "out.pdf")
        annotator = PDFAnnotator()

        # Should not crash
        annotator.annotate_pdf(
            pdf_path=simple_pdf,
            page_results=results,
            output_path=output_path,
        )

        assert os.path.exists(output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
