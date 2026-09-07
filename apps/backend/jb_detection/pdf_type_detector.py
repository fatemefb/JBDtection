"""JBDetection — PDF type detector.

Determines whether a PDF (or a specific page) is:

- ``"digital"`` — the PDF has extractable text (born-digital PDF, text
  layer present). No OCR is needed; we use PyMuPDF's native text
  extraction instead.
- ``"scanned"`` — the PDF is image-based (scanned document, or a
  digital PDF where the text is rasterised). PaddleOCR is required.

The detection is **per-page** — a single PDF can have mixed pages.
For convenience, :meth:`PdfTypeDetector.detect_pdf_type` samples a
few pages and returns a single verdict for the whole document.

Detection heuristic
-------------------
A page is "digital" if it has ``>= config.digital_pdf_min_text_chars``
characters of extractable text. The default threshold is 50 characters,
which is generous enough to handle pages with a few labels but strict
enough to catch truly scanned pages (which have zero extractable text).
"""

from __future__ import annotations

import enum
import logging
import os
from pathlib import Path
from typing import Optional

from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger("jb_detection.pdf_type_detector")


# ── Enum ───────────────────────────────────────────────────────────────
class PdfType(str, enum.Enum):
    """The extraction strategy for a PDF or page."""

    DIGITAL = "digital"    # Has extractable text → use native extraction
    SCANNED = "scanned"    # Image-based → use PaddleOCR
    EMPTY = "empty"        # No pages or unreadable
    UNKNOWN = "unknown"    # Detection failed


# ── Detector ──────────────────────────────────────────────────────────
class PdfTypeDetector:
    """Detect whether a PDF is digital (text-based) or scanned (image-based).

    The detector uses PyMuPDF (``fitz``) to check the amount of extractable
    text on each page. It does NOT render pages to images — that's expensive.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        if config is None:
            config = DEFAULT_CONFIG
        self._config = config

    # ── Per-page detection ─────────────────────────────────────────
    def detect_page_type(self, page: "object") -> PdfType:
        """Detect the type of a single PyMuPDF page.

        Parameters
        ----------
        page:
            A ``fitz.Page`` instance.

        Returns
        -------
        PdfType
            - :attr:`PdfType.DIGITAL` if the page has sufficient extractable
              text (≥ ``config.digital_pdf_min_text_chars``).
            - :attr:`PdfType.SCANNED` if the page has little or no text.
        """
        try:
            text = page.get_text("text")
        except Exception as exc:
            logger.debug("get_text failed: %s — treating as scanned", exc)
            return PdfType.SCANNED

        if not text:
            return PdfType.SCANNED

        text_len = len(text.strip())
        if text_len >= self._config.digital_pdf_min_text_chars:
            return PdfType.DIGITAL
        return PdfType.SCANNED

    # ── Per-PDF detection ──────────────────────────────────────────
    def detect_pdf_type(
        self,
        pdf_path: str,
        sample_pages: int = 5,
    ) -> PdfType:
        """Detect the overall type of a PDF by sampling pages.

        Parameters
        ----------
        pdf_path:
            Path to the PDF file.
        sample_pages:
            Number of pages to sample. If the PDF has fewer pages, all
            pages are checked. Defaults to 5.

        Returns
        -------
        PdfType
            The verdict for the whole PDF. If the majority of sampled
            pages are digital, the PDF is :attr:`PdfType.DIGITAL`.
            Otherwise it is :attr:`PdfType.SCANNED`.
        """
        p = Path(pdf_path)
        if not p.exists():
            return PdfType.UNKNOWN

        try:
            import fitz  # type: ignore
        except Exception as exc:
            logger.error("PyMuPDF not available: %s", exc)
            return PdfType.UNKNOWN

        try:
            doc = fitz.open(str(p))
        except Exception as exc:
            logger.error("Failed to open PDF %s: %s", p.name, exc)
            return PdfType.UNKNOWN

        try:
            page_count = len(doc)
            if page_count == 0:
                return PdfType.EMPTY

            # Sample pages: first, middle, last, plus a couple random.
            indices = self._sample_indices(page_count, sample_pages)

            digital_count = 0
            scanned_count = 0
            for idx in indices:
                try:
                    page = doc.load_page(idx)
                    ptype = self.detect_page_type(page)
                    if ptype == PdfType.DIGITAL:
                        digital_count += 1
                    elif ptype == PdfType.SCANNED:
                        scanned_count += 1
                except Exception as exc:
                    logger.debug("Failed to check page %d: %s", idx, exc)
                    scanned_count += 1  # Assume scanned on failure

            logger.info(
                "PDF type detection for %s: digital=%d, scanned=%d (of %d sampled)",
                p.name, digital_count, scanned_count, len(indices),
            )

            if digital_count > scanned_count:
                return PdfType.DIGITAL
            return PdfType.SCANNED
        finally:
            try:
                doc.close()
            except Exception:
                pass

    # ── Helper ────────────────────────────────────────────────────
    @staticmethod
    def _sample_indices(page_count: int, sample_size: int) -> list:
        """Pick up to ``sample_size`` page indices to inspect."""
        if page_count <= sample_size:
            return list(range(page_count))

        # Always include first, middle, last.
        indices = {0, page_count // 2, page_count - 1}
        # Fill remaining slots with evenly-spaced pages.
        step = page_count / sample_size
        for i in range(sample_size):
            indices.add(int(i * step))
        return sorted(indices)[:sample_size]


__all__ = ["PdfType", "PdfTypeDetector"]
