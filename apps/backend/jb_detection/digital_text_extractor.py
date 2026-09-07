"""JBDetection — digital PDF text extractor.

When a PDF is "digital" (has a text layer), we can extract text directly
via PyMuPDF without running OCR. This module converts PyMuPDF's text
extraction output into the same :class:`OcrDetection` objects that
:class:`TextDetector` produces, so the downstream pipeline
(:class:`PatternMatcher`, :class:`TagMatcher`) works identically
regardless of the extraction source.

Design
------
PyMuPDF's ``page.get_text("dict")`` returns a structured dict with
blocks → lines → spans. Each span has:
    {
        "text": "TE-5223",
        "bbox": (x0, y0, x1, y1),   # in PDF points (72 dpi)
        "size": 12.0,
        "font": "Helvetica",
        ...
    }

We convert each span (or a group of spans on the same line) into an
:class:`OcrDetection` with:
    - text: the span text
    - confidence: 1.0 (native extraction is exact — no OCR uncertainty)
    - polygon: 4-point polygon derived from the bbox
    - bbox: (x, y, width, height) in pixel coordinates

Coordinate conversion
----------------------
PyMuPDF gives coordinates in PDF points (72 dpi). The OCR pipeline works
in pixel coordinates at the render DPI. We convert:
    pixel = point * dpi / 72
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import Config, DEFAULT_CONFIG
from .models import OcrDetection

logger = logging.getLogger("jb_detection.digital_text_extractor")


# Confidence assigned to native (non-OCR) extraction.
# Native extraction is exact — there is no recognition uncertainty.
# We use 1.0 to distinguish from OCR confidence.
NATIVE_EXTRACTION_CONFIDENCE = 1.0


class DigitalTextExtractor:
    """Extract text from a digital (text-based) PDF page.

    Produces :class:`OcrDetection` objects identical in structure to what
    :class:`TextDetector` (PaddleOCR) produces, so downstream processing
    is identical.

    The extractor is **stateless** — it does not hold any model or
    reference data. It only needs a :class:`Config` for the DPI setting.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        if config is None:
            config = DEFAULT_CONFIG
        self._config = config

    # ── Public API ─────────────────────────────────────────────────
    def extract_from_page(
        self,
        page: "object",
        dpi: Optional[int] = None,
    ) -> List[OcrDetection]:
        """Extract all text spans from a single PyMuPDF page.

        Parameters
        ----------
        page:
            A ``fitz.Page`` instance.
        dpi:
            Render DPI for coordinate conversion. Defaults to
            ``config.pdf_dpi``. Coordinates in the returned
            :class:`OcrDetection` objects are in pixels at this DPI.

        Returns
        -------
        List[OcrDetection]
            One entry per text span, sorted in reading order
            (top→bottom, left→right).
        """
        if dpi is None:
            dpi = self._config.pdf_dpi

        try:
            text_dict = page.get_text("dict")
        except Exception as exc:
            logger.error("get_text('dict') failed: %s", exc)
            return []

        if not text_dict or "blocks" not in text_dict:
            return []

        scale = dpi / 72.0
        detections: List[OcrDetection] = []

        for block in text_dict.get("blocks", []):
            if block.get("type", 0) != 0:  # 0 = text block, 1 = image block
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    bbox_pts = span.get("bbox")  # (x0, y0, x1, y1) in points
                    if not bbox_pts or len(bbox_pts) != 4:
                        continue

                    detection = self._build_detection(text, bbox_pts, scale)
                    if detection is not None:
                        detections.append(detection)

        # Sort in reading order: top→bottom, left→right.
        detections.sort(key=lambda d: (d.bbox[1], d.bbox[0]))
        logger.debug("Extracted %d text spans from page (digital)", len(detections))
        return detections

    def extract_from_page_words(
        self,
        page: "object",
        dpi: Optional[int] = None,
    ) -> List[OcrDetection]:
        """Extract text as individual words (finer granularity).

        Uses ``page.get_text("words")`` which returns a list of
        ``(x0, y0, x1, y1, word, block_no, line_no, word_no)`` tuples.

        This is useful when spans contain multiple words that should be
        processed individually (e.g. "JB-101 JB-102" in one span).

        Parameters
        ----------
        page:
            A ``fitz.Page`` instance.
        dpi:
            Render DPI for coordinate conversion.

        Returns
        -------
        List[OcrDetection]
            One entry per word, sorted in reading order.
        """
        if dpi is None:
            dpi = self._config.pdf_dpi

        try:
            words = page.get_text("words")
        except Exception as exc:
            logger.error("get_text('words') failed: %s", exc)
            return []

        if not words:
            return []

        scale = dpi / 72.0
        detections: List[OcrDetection] = []

        for w in words:
            if len(w) < 5:
                continue
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
            text = str(text).strip()
            if not text:
                continue

            bbox_pts = (x0, y0, x1, y1)
            detection = self._build_detection(text, bbox_pts, scale)
            if detection is not None:
                detections.append(detection)

        detections.sort(key=lambda d: (d.bbox[1], d.bbox[0]))
        return detections

    # ── Internal ───────────────────────────────────────────────────
    def _build_detection(
        self,
        text: str,
        bbox_pts: Tuple[float, float, float, float],
        scale: float,
    ) -> Optional[OcrDetection]:
        """Build an :class:`OcrDetection` from a text span.

        Parameters
        ----------
        text:
            The extracted text.
        bbox_pts:
            ``(x0, y0, x1, y1)`` in PDF points.
        scale:
            Multiplier to convert points → pixels (``dpi / 72``).
        """
        x0, y0, x1, y1 = bbox_pts

        # Convert to pixel coordinates
        px_x0 = x0 * scale
        px_y0 = y0 * scale
        px_x1 = x1 * scale
        px_y1 = y1 * scale

        # Axis-aligned bbox: (x, y, width, height)
        x = int(round(px_x0))
        y = int(round(px_y0))
        w = max(1, int(round(px_x1 - px_x0)))
        h = max(1, int(round(px_y1 - px_y0)))

        # 4-point polygon (top-left, top-right, bottom-right, bottom-left)
        polygon = [
            [px_x0, px_y0],   # top-left
            [px_x1, px_y0],   # top-right
            [px_x1, px_y1],   # bottom-right
            [px_x0, px_y1],   # bottom-left
        ]

        return OcrDetection(
            text=text,
            confidence=NATIVE_EXTRACTION_CONFIDENCE,
            polygon=polygon,
            bbox=(x, y, w, h),
        )

    # ── Utility ────────────────────────────────────────────────────
    def extract_from_pdf(
        self,
        pdf_path: str,
        max_pages: Optional[int] = None,
    ) -> Dict[int, List[OcrDetection]]:
        """Extract text from all pages of a digital PDF.

        Parameters
        ----------
        pdf_path:
            Path to the PDF file.
        max_pages:
            Optional cap on the number of pages.

        Returns
        -------
        Dict[int, List[OcrDetection]]
            ``{page_number: [detections]}``. Page numbers are 1-indexed.
        """
        p = Path(pdf_path)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            import fitz  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                f"PyMuPDF (fitz) not available: {exc}. "
                f"Install with: pip install PyMuPDF"
            ) from exc

        doc = fitz.open(str(p))
        results: Dict[int, List[OcrDetection]] = {}
        try:
            page_count = len(doc)
            cap = max_pages if max_pages is not None else self._config.pdf_max_pages
            total = min(page_count, cap)

            for idx in range(total):
                try:
                    page = doc.load_page(idx)
                    detections = self.extract_from_page(page)
                    results[idx + 1] = detections
                except Exception as exc:
                    logger.error("Failed to extract page %d: %s", idx + 1, exc)
                    results[idx + 1] = []
        finally:
            try:
                doc.close()
            except Exception:
                pass

        return results


__all__ = ["DigitalTextExtractor", "NATIVE_EXTRACTION_CONFIDENCE"]
