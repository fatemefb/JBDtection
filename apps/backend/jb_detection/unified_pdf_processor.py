"""JBDetection — unified PDF processor.

This is the **single entry point** for PDF processing. It handles both
digital and scanned PDFs through one unified pipeline:

.. code-block:: text

                         PDF
                          │
                          ▼
                 Detect PDF Type (per page)
                    /          \\
                   /            \\
                  ▼              ▼
          Digital PDF       Scanned PDF
               │                    │
               │               PaddleOCR
               │                    │
               ▼                    ▼
       Native PDF Text       OCR Text + Boxes
       + Coordinates         + Confidence + Boxes
               │                    │
               └─────────┬──────────┘
                         ▼
                 Unified Detections  (List[OcrDetection])
                         │
                         ▼
                  PatternMatcher.match()
                         │
                         ▼
               JB / TAG / MC / Cable / SPARE
                         │
                         ▼
                  TagMatcher.match_tag()
                         │
                         ▼
                 JBDetectionResult

Key design principles
--------------------
1. **One OCR engine**: PaddleOCR is the only OCR engine. No Tesseract,
   no EasyOCR, no separate models for tables/diagrams.
2. **Digital PDF exception**: If a page has extractable text, we use
   PyMuPDF's native text extraction (no OCR). Both paths produce
   :class:`OcrDetection` objects.
3. **Unified detections**: :class:`PatternMatcher` receives the same
   data structure regardless of source. It has no knowledge of whether
   data came from OCR or native extraction.
4. **Extract everything first**: All text/regions are extracted from a
   page before pattern matching begins. No mid-extraction decisions.
5. **One OCR call per page**: PaddleOCR runs exactly once per scanned
   page. No repeated OCR for different categories (JB, Tag, etc.).
6. **Lazy processing**: Pages are processed one at a time. Images are
   released after processing. PaddleOCR is initialized once (singleton).
7. **No modes**: No ``table_mode``, ``diagram_mode``, or similar. The
   pipeline is truly unified.

Lifecycle
---------
- :class:`TextDetector` (PaddleOCR) is a **process-wide singleton** —
  initialized once on first use, reused for all subsequent pages.
- :class:`PatternMatcher` is stateful (holds regex patterns) and reused
  across all pages.
- :class:`TagMatcher` is stateful (holds IO List reference tags) and
  reused across all pages.
- :class:`DigitalTextExtractor` is stateless.
- :class:`PdfTypeDetector` is stateless.

The :class:`UnifiedPdfProcessor` holds all of these as collaborators.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .config import Config, DEFAULT_CONFIG
from .detector import TextDetector
from .digital_text_extractor import DigitalTextExtractor
from .image_preprocessor import preprocess, render_pdf_to_images
from .models import JBDetectionResult, OcrDetection, TagMatchInfo
from .pattern_matcher import PatternMatcher
from .pdf_type_detector import PdfType, PdfTypeDetector
from .tag_matcher import TagMatcher
from .progress import ProgressCallback, report_progress

logger = logging.getLogger("jb_detection.unified_pdf_processor")


class UnifiedPdfProcessor:
    """Unified PDF processor — handles both digital and scanned PDFs.

    This is the main entry point for PDF processing. It replaces the
    older :class:`JBDetectionPipeline.process_pdf` with a unified flow
    that automatically chooses between native text extraction (for
    digital PDFs) and PaddleOCR (for scanned PDFs).

    Parameters
    ----------
    config:
        :class:`Config` instance. Defaults to :data:`DEFAULT_CONFIG`.
    pattern_matcher:
        Optional pre-configured :class:`PatternMatcher`. If ``None``,
        a new one is created.
    tag_matcher:
        Optional pre-built :class:`TagMatcher`. If ``None``, tag
        matching against the IO List is skipped (detections are still
        classified by the :class:`PatternMatcher`).
    detector:
        Optional pre-initialized :class:`TextDetector`. If ``None``,
        the singleton is fetched lazily on first use.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        pattern_matcher: Optional[PatternMatcher] = None,
        tag_matcher: Optional[TagMatcher] = None,
        detector: Optional[TextDetector] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        if config is None:
            config = DEFAULT_CONFIG
        self._config = config
        self.progress_callback = progress_callback
        self._pattern_matcher = pattern_matcher or PatternMatcher()
        self._tag_matcher = tag_matcher  # may be None
        self._pattern_matcher.io_tag_matcher = self._tag_matcher
        self._detector = detector  # may be None — built lazily
        self._digital_extractor = DigitalTextExtractor(config=config)
        self._type_detector = PdfTypeDetector(config=config)

        # Stats
        self.pages_processed = 0
        self.pages_digital = 0
        self.pages_scanned = 0
        self.pages_failed = 0
        self.total_detections = 0
        self.total_tags = 0
        self.total_jbs = 0
        self.total_mcs = 0
        self.total_spares = 0
        self.processing_time = 0.0

        # Per-PDF type cache (for logging)
        self._pdf_types: Dict[str, PdfType] = {}

    # ── Properties ──────────────────────────────────────────────────
    @property
    def detector(self) -> TextDetector:
        """Lazy-fetch the :class:`TextDetector` singleton."""
        if self._detector is None:
            self._detector = TextDetector.get_instance(config=self._config)
        return self._detector

    @property
    def pattern_matcher(self) -> PatternMatcher:
        return self._pattern_matcher

    @property
    def tag_matcher(self) -> Optional[TagMatcher]:
        return self._tag_matcher

    @tag_matcher.setter
    def tag_matcher(self, value: Optional[TagMatcher]) -> None:
        self._tag_matcher = value
        self._pattern_matcher.io_tag_matcher = value

    def build_tag_matcher_from_excel(self, excel_path: str) -> TagMatcher:
        """Build (or rebuild) the :class:`TagMatcher` from an IO List."""
        self._tag_matcher = TagMatcher(config=self._config)
        self._tag_matcher.build_from_excel(excel_path)
        self._pattern_matcher.io_tag_matcher = self._tag_matcher
        return self._tag_matcher

    # ── Main entry: process_pdf ────────────────────────────────────
    def process_pdf(
        self,
        pdf_path: str,
        max_pages: Optional[int] = None,
    ) -> Dict[int, JBDetectionResult]:
        """Process every page of a PDF through the unified pipeline.

        For each page:
        1. Detect if the page is digital or scanned.
        2. If digital: extract text via PyMuPDF → ``List[OcrDetection]``.
        3. If scanned: render page → preprocess → PaddleOCR → ``List[OcrDetection]``.
        4. Feed detections to :class:`PatternMatcher` → :class:`JBDetectionResult`.
        5. If :class:`TagMatcher` is set, match tags against IO List.

        Parameters
        ----------
        pdf_path:
            Path to the PDF file.
        max_pages:
            Optional override for ``config.pdf_max_pages``.

        Returns
        -------
        Dict[int, JBDetectionResult]
            ``{page_number: result}``. Pages that fail to process
            return an empty :class:`JBDetectionResult`.
        """
        start_time = time.time()
        pdf_path = str(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Detect overall PDF type for logging
        overall_type = self._type_detector.detect_pdf_type(pdf_path)
        self._pdf_types[os.path.basename(pdf_path)] = overall_type
        logger.info(
            "Processing PDF: %s (overall type: %s)",
            os.path.basename(pdf_path), overall_type.value,
        )

        results: Dict[int, JBDetectionResult] = {}

        # Open the PDF once for digital extraction + type detection per page.
        # For scanned pages, we use render_pdf_to_images (which opens its own handle).
        try:
            import fitz  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                f"PyMuPDF (fitz) not available: {exc}"
            ) from exc

        doc = fitz.open(pdf_path)
        try:
            page_count = len(doc)
            cap = max_pages if max_pages is not None else self._config.pdf_max_pages
            total = min(page_count, cap)
            report_progress(self.progress_callback, "extract", pdf_path, 0, total)

            for idx in range(total):
                page_number = idx + 1
                try:
                    page = doc.load_page(idx)
                    page_type = self._type_detector.detect_page_type(page)

                    classified = None
                    if page_type == PdfType.DIGITAL:
                        detections = self._extract_digital(page)
                        if self._has_usable_text(detections):
                            classified = self._pattern_matcher.match(detections)
                        if classified is not None and (
                            classified.tags or classified.jb_identifiers
                            or classified.mc_identifiers or classified.spare_identifiers
                            or classified.cable_descriptions
                        ):
                            self.pages_digital += 1
                        else:
                            logger.info(
                                "Page %d: native extraction has no usable identifiers; falling back to OCR",
                                page_number,
                            )
                            self.total_detections -= len(detections)
                            page_type = PdfType.SCANNED
                            classified = None
                    if page_type != PdfType.DIGITAL:
                        self.pages_scanned += 1
                        detections = self._extract_scanned(
                            pdf_path, page_number, doc, idx,
                        )

                    result = self._process_detections(detections, page_number, classified)
                    results[page_number] = result

                except Exception as exc:
                    logger.error(
                        "Failed to process page %d of %s: %s",
                        page_number, os.path.basename(pdf_path), exc,
                    )
                    results[page_number] = JBDetectionResult()
                    self.pages_failed += 1

                finally:
                    report_progress(self.progress_callback, "extract", pdf_path, page_number, total)
                    # Periodic GC for large PDFs
                    if page_number % self._config.gc_interval_pages == 0:
                        gc.collect()

        finally:
            try:
                doc.close()
            except Exception:
                pass

        self.processing_time = time.time() - start_time
        self.pages_processed = len(results)
        logger.info(
            "PDF processed: %s — %d pages (digital=%d, scanned=%d, failed=%d), "
            "time=%.2fs",
            os.path.basename(pdf_path), len(results),
            self.pages_digital, self.pages_scanned, self.pages_failed,
            self.processing_time,
        )
        return results

    def process_multiple_pdfs(
        self,
        pdf_paths: List[str],
    ) -> Dict[str, Dict[int, JBDetectionResult]]:
        """Process several PDFs. Returns ``{pdf_name: {page: result}}``."""
        all_results: Dict[str, Dict[int, JBDetectionResult]] = {}
        for pdf_path in pdf_paths:
            pdf_name = os.path.basename(pdf_path)
            try:
                results = self.process_pdf(pdf_path)
                all_results[pdf_name] = results
            except Exception as exc:
                logger.error("Failed to process PDF %s: %s", pdf_path, exc)
                all_results[pdf_name] = {}
        return all_results

    # ── Per-page extraction paths ───────────────────────────────────
    @staticmethod
    def _has_usable_text(detections: List[OcrDetection]) -> bool:
        """Reject empty or broken text layers after native font decoding."""
        text = "".join(d.text for d in detections)
        if not text or not any(char.isalnum() for char in text):
            return False
        invalid = sum(
            char == "\ufffd" or (not char.isprintable() and not char.isspace())
            for char in text
        )
        return invalid / len(text) <= 0.05

    def _extract_digital(self, page: "object") -> List[OcrDetection]:
        """Extract text from a digital page via PyMuPDF.

        This is the "no-OCR" path. PyMuPDF's text extraction is
        instantaneous compared to OCR.
        """
        detections = self._digital_extractor.extract_from_page(page)
        matcher = self._tag_matcher
        detections = self._pattern_matcher._join_tag_fragments(detections)
        # A PDF span can contain several identifiers. Use actual word boxes
        # for such spans, preserving decoded spans if raw word text is unusable.
        words = None
        expanded = []
        for detection in detections:
            candidates = matcher.extract_candidates(detection.text) if matcher else []
            if candidates and not matcher.matches_io_pattern(detection.text):
                if words is None:
                    words = self._digital_extractor.extract_from_page_words(page)
                x, y, w, h = detection.bbox
                contained = [word for word in words
                    if x - 2 <= word.x and y - 2 <= word.y
                    and word.x + word.width <= x + w + 2
                    and word.y + word.height <= y + h + 2]
                grouped = []
                used = set()
                recovered = set()
                for candidate in candidates:
                    for start in range(len(contained)):
                        if start in used:
                            continue
                        for stop in range(start + 1, min(len(contained), start + 8) + 1):
                            if any(index in used for index in range(start, stop)):
                                break
                            group = contained[start:stop]
                            text = " ".join(word.text for word in group)
                            if matcher.separator_key(text) == matcher.separator_key(candidate):
                                grouped.append(self._combine_detections(group, candidate))
                                used.update(range(start, stop))
                                recovered.add(candidate)
                                break
                        if candidate in recovered:
                            break
                if contained and set(candidates).issubset(recovered):
                    grouped.extend(word for index, word in enumerate(contained) if index not in used)
                    expanded.extend(sorted(grouped, key=lambda word: (word.y, word.x)))
                    continue
            expanded.append(detection)
        detections = expanded
        logger.debug(
            "Digital extraction: %d detections", len(detections),
        )
        self.total_detections += len(detections)
        return detections

    @staticmethod
    def _combine_detections(detections: List[OcrDetection], text: str) -> OcrDetection:
        x = min(item.x for item in detections)
        y = min(item.y for item in detections)
        right = max(item.x + item.width for item in detections)
        bottom = max(item.y + item.height for item in detections)
        return OcrDetection(text, min(item.confidence for item in detections), [], (x, y, right - x, bottom - y))

    def _extract_scanned(
        self,
        pdf_path: str,
        page_number: int,
        doc: "object",
        page_idx: int,
    ) -> List[OcrDetection]:
        """Extract text from a scanned page via PaddleOCR.

        This is the "OCR" path. We render the page to a BGR image,
        preprocess it, and run PaddleOCR exactly once.
        """
        # Render the specific page to an image.
        # We use the existing render_pdf_to_images function but it
        # renders from the start. For a single page, we use load_pdf_page.
        from .image_preprocessor import load_pdf_page

        try:
            page = doc.load_page(page_idx)
            # Compute effective DPI (same logic as render_pdf_to_images)
            page_count = len(doc)
            from .image_preprocessor import _compute_effective_dpi
            effective_dpi = _compute_effective_dpi(self._config, page_count)
            image = load_pdf_page(page, dpi=effective_dpi)
        except Exception as exc:
            raise RuntimeError(f"Failed to render page {page_number}: {exc}") from exc

        if image is None or image.size == 0:
            raise RuntimeError(f"Failed to render page {page_number}: empty image")

        try:
            # Preprocess (CLAHE + Otsu) — same as the OCR pipeline
            preprocessed = preprocess(image, config=self._config)
        except Exception as exc:
            logger.warning("Preprocess failed on page %d: %s — using raw image", page_number, exc)
            preprocessed = image

        try:
            # PaddleOCR — exactly ONE call per page
            detections = self.detector.detect(preprocessed)
        except Exception as exc:
            raise RuntimeError(f"OCR failed on page {page_number}: {exc}") from exc

        self.total_detections += len(detections)
        return detections

    # ── Pattern matching + tag matching ────────────────────────────
    def _process_detections(
        self,
        detections: List[OcrDetection],
        page_number: int,
        classified: Optional[JBDetectionResult] = None,
    ) -> JBDetectionResult:
        """Run pattern matching + tag matching on a list of detections.

        This is the unified post-extraction step. It does NOT know or
        care whether detections came from OCR or native extraction.
        """
        # 1. Classify detections → JBDetectionResult
        result = classified if classified is not None else self._pattern_matcher.match(detections)

        # 2. Match tags against IO List (if a TagMatcher is set)
        if self._tag_matcher is not None and result.tags:
            for tag in list(result.tags):
                match_type, score, matched_tag = self._tag_matcher.match_tag(tag)
                info = result.tag_match_info.get(tag) or TagMatchInfo(
                    ocr_text=tag, bbox=(0, 0, 0, 0),
                )
                info.match_type = match_type
                info.score = score
                info.matched_tag = matched_tag
                if match_type == "exact":
                    info.reason = f"Exact IO List match: {matched_tag}"
                elif match_type == "similar":
                    info.reason = f"Not in IO List; closest tag: {matched_tag} ({score:.1%}). Review required."
                else:
                    info.reason = f"Matches learned IO List pattern but is absent; closest tag: {matched_tag or 'none'} ({score:.1%}). Review required."
                result.tag_match_info[tag] = info

        # 3. Update stats
        self.total_tags += len(result.tags)
        self.total_jbs += len(result.jb_identifiers)
        self.total_mcs += len(result.mc_identifiers)
        self.total_spares += len(result.spare_identifiers)

        logger.info(
            "Page %d processed: %d detections → %d tags, %d JBs, %d MCs, %d spares",
            page_number, len(detections), len(result.tags),
            len(result.jb_identifiers), len(result.mc_identifiers),
            len(result.spare_identifiers),
        )

        return result

    # ── Stats ──────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        """Return processing statistics."""
        return {
            "pages_processed": self.pages_processed,
            "pages_digital": self.pages_digital,
            "pages_scanned": self.pages_scanned,
            "pages_failed": self.pages_failed,
            "total_detections": self.total_detections,
            "total_tags": self.total_tags,
            "total_jbs": self.total_jbs,
            "total_mcs": self.total_mcs,
            "total_spares": self.total_spares,
            "processing_time": round(self.processing_time, 2),
            "detector_stats": self.detector.stats() if self._detector else {},
            "tag_matcher_stats": (
                self._tag_matcher.stats() if self._tag_matcher else {}
            ),
            "pdf_types": {k: v.value for k, v in self._pdf_types.items()},
        }

    def reset_stats(self) -> None:
        """Reset all counters to zero."""
        self.pages_processed = 0
        self.pages_digital = 0
        self.pages_scanned = 0
        self.pages_failed = 0
        self.total_detections = 0
        self.total_tags = 0
        self.total_jbs = 0
        self.total_mcs = 0
        self.total_spares = 0
        self.processing_time = 0.0
        self._pdf_types.clear()


__all__ = ["UnifiedPdfProcessor"]
