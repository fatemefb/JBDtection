"""JBDetection — PDF annotator.

Generates an annotated PDF with color-coded bounding boxes for each
detected category (Tag, JB, MC, Cable, SPARE).

Color scheme (BGR):
    - TAG     → green   (0, 200, 0)
    - JB      → blue    (255, 100, 0)
    - MC      → orange  (0, 165, 255)
    - CABLE   → yellow  (0, 255, 255)
    - SPARE   → gray    (128, 128, 128)
    - UNKNOWN → red     (0, 0, 255)

The annotator reads :class:`JBDetectionResult` objects (which contain
:class:`TagMatchInfo` with bbox coordinates) and draws rectangles on
the original PDF pages using PyMuPDF's drawing API.

Coordinate system
-----------------
Bounding boxes in :class:`OcrDetection` / :class:`TagMatchInfo` are in
**pixel** coordinates (relative to the rendered page image at a given
DPI). PyMuPDF's drawing API uses **PDF point** coordinates (72 DPI).
Conversion: ``point = pixel * 72 / dpi``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import Config, DEFAULT_CONFIG
from .models import JBDetectionResult, TagMatchInfo

logger = logging.getLogger("jb_detection.annotator")


# ── Category colors (RGB for PyMuPDF; converted from BGR config) ──────
# PyMuPDF uses RGB tuples with float values in [0, 1] for shape drawing,
# while OpenCV uses BGR with int values in [0, 255].
# Our config stores BGR (for OpenCV compatibility), so we convert here:
# BGR int → RGB float (normalize to [0, 1]).
def _bgr_to_rgb_float(bgr: tuple) -> tuple:
    """Convert BGR int (0-255) to RGB float (0.0-1.0) for PyMuPDF."""
    return (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)


CATEGORY_COLORS_RGB = {
    "tag":    _bgr_to_rgb_float((0, 200, 0)),    # green
    "jb":     _bgr_to_rgb_float((255, 100, 0)),  # blue
    "mc":     _bgr_to_rgb_float((0, 165, 255)),  # orange
    "cable":  _bgr_to_rgb_float((0, 255, 255)),  # yellow
    "spare":  _bgr_to_rgb_float((128, 128, 128)),# gray
    "unknown":_bgr_to_rgb_float((0, 0, 255)),    # red
}

# Default DPI for coordinate conversion when render DPI is unknown.
_DEFAULT_DPI = 300


class PDFAnnotator:
    """Generate an annotated PDF with color-coded bounding boxes.

    The annotator reads per-page :class:`JBDetectionResult` objects and
    draws rectangles on the original PDF. Each category gets a distinct
    color.

    Usage
    -----
    ::

        annotator = PDFAnnotator(config=config)
        counts = annotator.annotate_pdf(
            pdf_path="input.pdf",
            page_results={1: result_page1, 2: result_page2},
            output_path="annotated_output.pdf",
            tag_to_number={"TE-5223": 1, "PT-1014": 2},
            all_pdf_results={"input.pdf": {1: result_page1, 2: result_page2}},
            pdf_name="input.pdf",
        )
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        if config is None:
            config = DEFAULT_CONFIG
        self._config = config
        # Track the render DPI per page for coordinate conversion.
        # If not provided, we assume config.pdf_dpi.
        self._page_dpi: Dict[int, int] = {}

    def annotate_pdf(
        self,
        pdf_path: str,
        page_results: Dict[int, JBDetectionResult],
        output_path: str,
        tag_to_number: Optional[Dict[str, int]] = None,
        all_pdf_results: Optional[Dict[str, Dict[int, JBDetectionResult]]] = None,
        pdf_name: Optional[str] = None,
        render_dpi: Optional[int] = None,
    ) -> Dict[str, int]:
        """Annotate a PDF with color-coded bounding boxes.

        Parameters
        ----------
        pdf_path:
            Path to the original PDF.
        page_results:
            ``{page_number: JBDetectionResult}`` — per-page detection
            results. Page numbers are 1-indexed.
        output_path:
            Where to write the annotated PDF.
        tag_to_number:
            Master tag → number mapping (for labeling). Optional.
        all_pdf_results:
            All PDFs' results (for cross-PDF duplicate detection labels).
            Optional — used only for logging.
        pdf_name:
            Filename of the PDF (for logging). Optional.
        render_dpi:
            The DPI at which pages were rendered when OCR'd. If ``None``,
            uses ``config.pdf_dpi``. This is needed to convert pixel
            coordinates back to PDF point coordinates for drawing.

        Returns
        -------
        Dict[str, int]
            Counts of annotations drawn per category:
            ``{"tags": N, "jbs": N, "mcs": N, "cables": N, "spares": N}``.
        """
        p = Path(pdf_path)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if render_dpi is None:
            render_dpi = self._config.pdf_dpi

        try:
            import fitz  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                f"PyMuPDF (fitz) not available: {exc}. "
                f"Install with: pip install PyMuPDF"
            ) from exc

        tag_to_number = tag_to_number or {}
        counts = {"tags": 0, "jbs": 0, "mcs": 0, "cables": 0, "spares": 0}

        doc = fitz.open(str(p))
        try:
            page_count = len(doc)
            scale = 72.0 / render_dpi  # pixel → point conversion

            for page_number, result in page_results.items():
                if page_number < 1 or page_number > page_count:
                    logger.warning(
                        "Page %d out of range (1-%d) — skipping", page_number, page_count
                    )
                    continue

                try:
                    page = doc.load_page(page_number - 1)  # 0-indexed
                    page_counts = self._annotate_page(
                        page, result, tag_to_number, scale,
                    )
                    for k, v in page_counts.items():
                        counts[k] = counts.get(k, 0) + v
                except Exception as exc:
                    logger.error(
                        "Failed to annotate page %d of %s: %s",
                        page_number, p.name, exc,
                    )

            # Save
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            doc.save(output_path, garbage=3, deflate=True)
            logger.info(
                "Annotated PDF saved: %s (tags=%d, jbs=%d, mcs=%d, cables=%d, spares=%d)",
                output_path, counts["tags"], counts["jbs"],
                counts["mcs"], counts["cables"], counts["spares"],
            )
        finally:
            try:
                doc.close()
            except Exception:
                pass

        return counts

    # ── Per-page annotation ────────────────────────────────────────
    def _annotate_page(
        self,
        page: "object",
        result: JBDetectionResult,
        tag_to_number: Dict[str, int],
        scale: float,
    ) -> Dict[str, int]:
        """Draw bounding boxes on a single page.

        Returns counts per category.
        """
        import fitz  # type: ignore

        counts = {"tags": 0, "jbs": 0, "mcs": 0, "cables": 0, "spares": 0}

        # ── Tags (green) ────────────────────────────────────────────
        for tag, info in result.tag_match_info.items():
            bbox = info.bbox
            if bbox and bbox != (0, 0, 0, 0):
                rect = self._bbox_to_rect(bbox, scale)
                self._draw_box(page, rect, CATEGORY_COLORS_RGB["tag"],
                               label=self._tag_label(tag, info, tag_to_number))
                counts["tags"] += 1

        # ── JBs (blue) ─────────────────────────────────────────────
        for jb in result.jb_identifiers:
            pos = result.tag_positions.get(jb)
            if pos:
                rect = self._pos_to_rect(pos, scale)
                self._draw_box(page, rect, CATEGORY_COLORS_RGB["jb"], label=str(jb))
                counts["jbs"] += 1

        # ── MCs (orange) ──────────────────────────────────────────
        for mc in result.mc_identifiers:
            pos = result.tag_positions.get(mc)
            if pos:
                rect = self._pos_to_rect(pos, scale)
                self._draw_box(page, rect, CATEGORY_COLORS_RGB["mc"], label=str(mc))
                counts["mcs"] += 1

        # ── Cables (yellow) ────────────────────────────────────────
        for cable in result.cable_descriptions:
            # Cables don't have positions in tag_positions; use tag_positions
            # if available, otherwise skip drawing (text-only).
            pos = result.tag_positions.get(cable)
            if pos:
                rect = self._pos_to_rect(pos, scale)
                self._draw_box(page, rect, CATEGORY_COLORS_RGB["cable"], label="CABLE")
                counts["cables"] += 1

        # ── SPAREs (gray) ──────────────────────────────────────────
        for spare in result.spare_identifiers:
            # Find position from spare_positions if available
            spare_pos = None
            for sp in result.spare_positions:
                if isinstance(sp, dict) and sp.get("text", "").upper() == str(spare).upper():
                    spare_pos = sp.get("position")
                    break
            if spare_pos:
                rect = self._pos_to_rect(spare_pos, scale)
                self._draw_box(page, rect, CATEGORY_COLORS_RGB["spare"], label=str(spare))
                counts["spares"] += 1

        return counts

    # ── Drawing helpers ────────────────────────────────────────────
    def _bbox_to_rect(
        self,
        bbox: Tuple[int, int, int, int],
        scale: float,
    ) -> "fitz.Rect":
        """Convert a pixel bbox (x, y, w, h) to a PyMuPDF Rect in points."""
        import fitz  # type: ignore
        x, y, w, h = bbox
        x0 = x * scale
        y0 = y * scale
        x1 = (x + w) * scale
        y1 = (y + h) * scale
        return fitz.Rect(x0, y0, x1, y1)

    def _pos_to_rect(
        self,
        pos: Tuple[int, int],
        scale: float,
        default_size: int = 80,
    ) -> "fitz.Rect":
        """Convert a position (x, y) to a small Rect in points.

        Used when only a position is known (not a full bbox). We draw
        a small rectangle around the position.
        """
        import fitz  # type: ignore
        x, y = pos[0], pos[1]
        x0 = x * scale
        y0 = y * scale
        x1 = (x + default_size) * scale
        y1 = (y + default_size) * scale
        return fitz.Rect(x0, y0, x1, y1)

    def _draw_box(
        self,
        page: "object",
        rect: "fitz.Rect",
        color: tuple,
        label: Optional[str] = None,
    ) -> None:
        """Draw a colored rectangle + optional label on the page."""
        import fitz  # type: ignore

        # Draw the rectangle (1.5 point border, no fill)
        page.draw_rect(
            rect,
            color=color,
            fill=None,
            width=1.5,
            overlay=True,
        )

        # Draw the label above the box
        if label:
            label_point = fitz.Point(rect.x0, rect.y0 - 2)
            try:
                page.insert_text(
                    label_point,
                    str(label)[:50],  # Truncate long labels
                    fontsize=6,
                    color=color,
                    overlay=True,
                )
            except Exception as exc:
                logger.debug("Failed to insert label '%s': %s", label, exc)

    def _tag_label(
        self,
        tag: str,
        info: TagMatchInfo,
        tag_to_number: Dict[str, int],
    ) -> str:
        """Build a label for a tag annotation."""
        parts = [str(tag)]
        num = tag_to_number.get(tag) or tag_to_number.get(info.matched_tag)
        if num:
            parts.append(f"#{num}")
        if info.match_type == "exact":
            parts.append("[E]")
        elif info.match_type == "similar":
            parts.append(f"[S:{info.score:.2f}]")
        return " ".join(parts)


__all__ = ["PDFAnnotator", "CATEGORY_COLORS_RGB"]
