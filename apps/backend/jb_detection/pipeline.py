"""JBDetection — unified processing pipeline.

The pipeline is **single and unified** — there is no separate
"table mode" vs "diagram mode". The same five steps run for every page:

1. **Preprocess** the rasterized page (CLAHE + Otsu — see
   :mod:`image_preprocessor`).
2. **Detect** text via PaddleOCR (see :mod:`detector`).
3. **Match patterns** to classify detections as JB / MC / Tag / Cable /
   SPARE (see :mod:`pattern_matcher`).
4. **Match tags** against the IO List (exact / fuzzy / unmatched — see
   :mod:`tag_matcher`).
5. **Assign tag numbers** by vertical position (top-to-bottom).

The result is a :class:`JBDetectionResult` per page, exposing the
legacy 9-tuple via :meth:`JBDetectionResult.to_tuple` for backward
compatibility with the facade.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .config import Config
from .detector import TextDetector
from .image_preprocessor import preprocess, render_pdf_to_images
from .models import JBDetectionResult, OcrDetection, PageResult, TagMatchInfo
from .pattern_matcher import PatternMatcher
from .tag_matcher import TagMatcher

logger = logging.getLogger("jb_detection.pipeline")


class JBDetectionPipeline:
    """Single unified pipeline — no mode split.

    The pipeline holds three collaborators:

    - :class:`TextDetector` (PaddleOCR wrapper, singleton)
    - :class:`PatternMatcher` (stateful regex classifier)
    - :class:`TagMatcher` (IO-List fuzzy matcher, optional)

    A single :class:`PatternMatcher` is reused across all pages.
    The :class:`TagMatcher` is built lazily — only when an IO List is
    provided.
    """

    def __init__(self,
                 config: Optional[Config] = None,
                 pattern_matcher: Optional[PatternMatcher] = None,
                 tag_matcher: Optional[TagMatcher] = None,
                 detector: Optional[TextDetector] = None) -> None:
        if config is None:
            from .config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG
        self._config = config
        self._pattern_matcher = pattern_matcher or PatternMatcher()
        self._tag_matcher = tag_matcher  # may be None — set later
        self._detector = detector  # may be None — built lazily

        # Stats
        self.pages_processed = 0
        self.total_detections = 0
        self.total_tags = 0
        self.total_jbs = 0
        self.total_mcs = 0
        self.total_spares = 0

    # ── Configuration setters ──────────────────────────────────────
    @property
    def pattern_matcher(self) -> PatternMatcher:
        return self._pattern_matcher

    @property
    def tag_matcher(self) -> Optional[TagMatcher]:
        return self._tag_matcher

    @tag_matcher.setter
    def tag_matcher(self, value: Optional[TagMatcher]) -> None:
        self._tag_matcher = value

    @property
    def detector(self) -> TextDetector:
        if self._detector is None:
            self._detector = TextDetector.get_instance(config=self._config)
        return self._detector

    def build_tag_matcher_from_excel(self, excel_path: str) -> TagMatcher:
        """Build (or rebuild) the :class:`TagMatcher` from an IO List."""
        self._tag_matcher = TagMatcher(config=self._config)
        self._tag_matcher.build_from_excel(excel_path)
        return self._tag_matcher

    # ── Single-page entry point ────────────────────────────────────
    def process_page(self,
                      image: Any,
                      page_number: int = 1,
                      ) -> JBDetectionResult:
        """Run the unified pipeline on a single page image.

        Parameters
        ----------
        image:
            BGR numpy array (HxWx3, uint8) — typically the output of
            :func:`render_pdf_to_images`.
        page_number:
            1-indexed page number — used for logging only.

        Returns
        -------
        JBDetectionResult
        """
        if image is None or image.size == 0:
            return JBDetectionResult()

        # 1. Preprocess (UNIFIED — no mode split)
        try:
            preprocessed = preprocess(image, config=self._config)
        except Exception as exc:
            logger.error("Preprocess failed on page %d: %s", page_number, exc)
            preprocessed = image

        # 2. Detect text via PaddleOCR
        try:
            detections = self.detector.detect(preprocessed)
        except Exception as exc:
            logger.error("OCR failed on page %d: %s", page_number, exc)
            detections = []

        self.total_detections += len(detections)

        # 3. Classify detections
        result = self._pattern_matcher.match(detections)

        # 4. Match tags against IO List (if a TagMatcher is set)
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
                    info.reason = f"Fuzzy match (score={score:.3f}): {matched_tag}"
                else:
                    info.reason = "No IO List match"
                result.tag_match_info[tag] = info

        # 5. Stats
        self.pages_processed += 1
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

    # ── Single-PDF entry point ────────────────────────────────────
    def process_pdf(self,
                     pdf_path: str,
                     max_pages: Optional[int] = None,
                     ) -> Dict[int, JBDetectionResult]:
        """Run the pipeline on every page of a PDF.

        Returns
        -------
        Dict[int, JBDetectionResult]
            ``{page_number: result}``. Pages that fail to render are
            silently skipped (with a log message).
        """
        pdf_path = str(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        results: Dict[int, JBDetectionResult] = {}
        for page_number, image, w, h in render_pdf_to_images(
            pdf_path, config=self._config, max_pages=max_pages,
        ):
            try:
                result = self.process_page(image, page_number=page_number)
                results[page_number] = result
            except Exception as exc:
                logger.error("Pipeline failed on page %d: %s", page_number, exc)
                results[page_number] = JBDetectionResult()
            finally:
                # Free the page image before the next iteration
                del image
                if page_number % self._config.gc_interval_pages == 0:
                    gc.collect()
        return results

    # ── Multi-PDF entry point ─────────────────────────────────────
    def process_multiple_pdfs(self,
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

    # ── Stats ──────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        return {
            "pages_processed": self.pages_processed,
            "total_detections": self.total_detections,
            "total_tags": self.total_tags,
            "total_jbs": self.total_jbs,
            "total_mcs": self.total_mcs,
            "total_spares": self.total_spares,
            "detector_stats": self.detector.stats() if self._detector else {},
            "tag_matcher_stats": (self._tag_matcher.stats()
                                    if self._tag_matcher else {}),
        }

    def reset_stats(self) -> None:
        self.pages_processed = 0
        self.total_detections = 0
        self.total_tags = 0
        self.total_jbs = 0
        self.total_mcs = 0
        self.total_spares = 0


__all__ = ["JBDetectionPipeline"]
