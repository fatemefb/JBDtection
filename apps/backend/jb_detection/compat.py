"""JBDetection — compatibility layer for legacy ``DataAnalysisModule``.

This module provides drop-in replacements for the old
``DataAnalysisModule.DataAnalysis`` and ``DataAnalysisModule.TagJBExtractor``
classes. It is designed so that ``apps/backend/app.py`` and its companion
files (``TagJBExtractorLogger.py``, ``LinuxTagJBExtractor.py``,
``LinuxTagJBExtractorLogger.py``) can switch to the new PaddleOCR-based
pipeline by changing **one import line**.

Usage
-----
Replace in ``app.py``::

    from DataAnalysisModule import DataAnalysis, TagJBExtractor

with::

    from jb_detection.compat import DataAnalysis, TagJBExtractor

and replace in ``TagJBExtractorLogger.py``::

    from DataAnalysisModule import TagJBExtractor

with::

    from jb_detection.compat import TagJBExtractor

Same for ``LinuxTagJBExtractor.py``.

Backward-compatible API surface
-------------------------------
The legacy ``TagJBExtractor`` exposed many methods that the rest of the
code base calls. This compatibility wrapper implements all of them by
delegating to the new :class:`jb_detection.facade.TagJBExtractor` and
:class:`jb_detection.unified_pdf_processor.UnifiedPdfProcessor`.

Legacy methods provided:

- ``__init__(tesseract_path=None, excel_path=None)`` (tesseract_path ignored)
- ``set_patterns(jb_examples, mc_examples, spare_examples, cable_examples, ...)``
- ``set_terminal_wire_patterns(config_dict)``
- ``set_wire_color_rule(rule)``
- ``set_scr_number_rule(rule)``
- ``build_tag_vectors_from_excel(excel_path)``
- ``extract_from_image(image)`` → legacy 9-tuple
- ``process_pdf(pdf_path)`` → Dict[int, 9-tuple]
- ``process_multiple_pdfs(pdf_paths)`` → Dict[str, Dict[int, 9-tuple]]
- ``run(pdf_paths, excel_path, output_excel_path, intermediate_excel_path, all_ocr_tags)``
- ``run_with_annotated_pdf(pdf_paths, excel_path, output_excel_path, output_pdf_dir)``
- ``create_annotated_pdf(pdf_path, output_pdf_path)``
- ``_create_unmatched_tags_excel(unmatched_excel_tags, unmatched_pdf_tags, output_path)``
- ``get_processing_stats()``
- ``reset_stats()``
- ``generate_scr_number(tag_number)``
- ``generate_terminal_numbers(tag_number)``
- ``generate_mc_wire_colors(tag_number)``
- ``generate_mc_wire_colors_enhanced(tag_number)``
- ``add_wire_colors_and_scr_to_dataframe(df, tag_to_number, output_path, pdf_results, io_tags, pdf_name)``
- ``extract_pair_number(cable_description)``
- ``check_tag_number_consistency(tag_to_number)``
- ``gpu_available`` (property)
- ``gpu_type`` (property)
- ``cuda_device_count`` (property)
- ``latest_pattern_unmatched_candidates`` (attribute)
- ``latest_pattern_unmatched_details`` (attribute)
- ``latest_warnings`` (attribute)
- ``all_tags``, ``matched_tags``, ``all_jbs``, ``all_mcs``, ``all_spares``
- ``exact_matches``, ``similar_matches``, ``processing_time``
- ``document_types``, ``document_type_by_path``, ``page_results``
- ``tag_patterns``, ``jb_regex``, ``mc_regex``, ``spare_regex``
- ``vector_matcher`` (alias for the internal tag matcher)
- ``set_classifier(classifier)`` (no-op, kept for compat)

Legacy methods NOT in this list (rarely used) will raise ``AttributeError``
with a helpful message.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# ── Import the new pipeline ───────────────────────────────────────────
from .config import Config, DEFAULT_CONFIG, load_config
from .detector import TextDetector
from .excel_exporter import ExcelExporter
from .facade import TagJBExtractor as _NewTagJBExtractor
from .facade import DataAnalysis as _NewDataAnalysis
from .models import JBDetectionResult, TagMatchInfo
from .pattern_matcher import PatternMatcher
from .pipeline import JBDetectionPipeline
from .tag_matcher import TagMatcher
from .unified_pdf_processor import UnifiedPdfProcessor

logger = logging.getLogger("jb_detection.compat")


# ═══════════════════════════════════════════════════════════════════════════
# GPU detection (re-implemented to avoid importing tensorflow)
# ═══════════════════════════════════════════════════════════════════════════
def _detect_gpu() -> Tuple[bool, str, int]:
    """Detect GPU availability without depending on TensorFlow.

    Returns
    -------
    (gpu_available: bool, gpu_type: str, cuda_device_count: int)
    """
    gpu_available = False
    gpu_type = "None"
    cuda_device_count = 0

    # Check PaddlePaddle CUDA first (lightweight, no subprocess)
    try:
        import paddle  # type: ignore
        if paddle.is_compiled_with_cuda():
            gpu_available = True
            gpu_type = "NVIDIA"
            try:
                cuda_device_count = paddle.device.cuda.device_count()
            except Exception:
                cuda_device_count = 1
            return (gpu_available, gpu_type, cuda_device_count)
    except Exception:
        pass

    # Fall back to nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_available = True
            gpu_type = "NVIDIA"
            cuda_device_count = 1  # conservative default
            return (gpu_available, gpu_type, cuda_device_count)
    except Exception:
        pass

    return (gpu_available, gpu_type, cuda_device_count)


# Cache GPU detection result (runs once per process)
_GPU_CACHE: Optional[Tuple[bool, str, int]] = None
_GPU_LOCK = threading.Lock()


def _get_gpu_info() -> Tuple[bool, str, int]:
    """Return cached GPU info, detecting on first call."""
    global _GPU_CACHE
    if _GPU_CACHE is None:
        with _GPU_LOCK:
            if _GPU_CACHE is None:
                _GPU_CACHE = _detect_gpu()
                logger.info(
                    "GPU detection: available=%s, type=%s, cuda_devices=%d",
                    _GPU_CACHE[0], _GPU_CACHE[1], _GPU_CACHE[2],
                )
    return _GPU_CACHE


# ═══════════════════════════════════════════════════════════════════════════
# Compat TagJBExtractor
# ═══════════════════════════════════════════════════════════════════════════
class TagJBExtractor:
    """Drop-in replacement for ``DataAnalysisModule.TagJBExtractor``.

    Internally uses the new :class:`UnifiedPdfProcessor` (PaddleOCR +
    native digital extraction) but preserves the full legacy API.

    Constructor accepts the same arguments as the original so that
    existing call sites (``TagJBExtractor(tesseract_path=..., excel_path=...)``)
    work unchanged. ``tesseract_path`` is accepted but silently ignored
    because the new pipeline uses PaddleOCR.
    """

    def __init__(
        self,
        tesseract_path: Optional[str] = None,
        excel_path: Optional[str] = None,
    ) -> None:
        # ── Ignore tesseract_path (legacy, kept for API compat) ──
        if tesseract_path:
            logger.info(
                "tesseract_path=%s ignored (jb_detection uses PaddleOCR)",
                tesseract_path,
            )

        # ── Configuration ─────────────────────────────────────────
        self._config: Config = load_config()

        # ── Core collaborators ────────────────────────────────────
        self._pattern_matcher = PatternMatcher()
        self._tag_matcher = TagMatcher(config=self._config)
        self._excel_exporter = ExcelExporter(config=self._config)
        self._excel_exporter.set_pattern_matcher(self._pattern_matcher)

        # The unified processor replaces the old TagJBExtractor pipeline.
        self._processor = UnifiedPdfProcessor(
            config=self._config,
            pattern_matcher=self._pattern_matcher,
            tag_matcher=self._tag_matcher,
        )

        # ── Legacy state attributes ───────────────────────────────
        self.excel_path: Optional[str] = excel_path
        self.all_tags: Set[str] = set()
        self.matched_tags: Set[str] = set()
        self.all_jbs: Set[str] = set()
        self.all_mcs: Set[str] = set()
        self.all_spares: List[str] = []
        self.exact_matches = 0
        self.similar_matches = 0
        self.processing_time = 0.0
        self.latest_pattern_unmatched_candidates: List[str] = []
        self.latest_pattern_unmatched_details: List[Dict[str, Any]] = []
        self.latest_warnings: List[Dict[str, Any]] = []
        self._page_warnings: List[Dict[str, Any]] = []

        # Backward-compatible attributes expected by external callers
        self.jb_examples: Optional[str] = None
        self.mc_examples: Optional[str] = None
        self.spare_examples: Optional[str] = None
        self.jb_examples_list: List[str] = []
        self.mc_examples_list: List[str] = []
        self.spare_examples_list: List[str] = []
        self.cable_examples: Optional[str] = None
        self.wire_color_rule: Optional[str] = None
        self.scr_number_rule: Optional[str] = None
        self.terminal_pattern: Optional[str] = None
        self.terminal_pattern_dict: Dict[str, Any] = {}
        self.jb_regex = self._pattern_matcher.jb_regex
        self.mc_regex = self._pattern_matcher.mc_regex
        self.spare_regex = self._pattern_matcher.spare_regex
        self.vector_matcher = self._tag_matcher  # backward-compatible alias
        self.tag_patterns: Dict[str, Any] = {}
        self._classifier = None
        self._current_pdf_type = "diagrams"
        self.document_types: Dict[str, str] = {}
        self.document_type_by_path: Dict[str, str] = {}
        self.document_nature_by_path: Dict[str, str] = {}
        self.page_results: Dict[int, Any] = {}

        # ── GPU info (lazy) ────────────────────────────────────────
        self._gpu_info: Optional[Tuple[bool, str, int]] = None

        # ── If an excel_path was supplied, eagerly build the tag matcher ──
        if excel_path and os.path.exists(excel_path):
            try:
                self.build_tag_vectors_from_excel(excel_path)
            except Exception as exc:
                logger.warning(
                    "Failed to build tag vectors from %s: %s", excel_path, exc,
                )

    # ── GPU properties (legacy API) ───────────────────────────────
    @property
    def gpu_available(self) -> bool:
        if self._gpu_info is None:
            self._gpu_info = _get_gpu_info()
        return self._gpu_info[0]

    @property
    def gpu_type(self) -> str:
        if self._gpu_info is None:
            self._gpu_info = _get_gpu_info()
        return self._gpu_info[1]

    @property
    def cuda_device_count(self) -> int:
        if self._gpu_info is None:
            self._gpu_info = _get_gpu_info()
        return self._gpu_info[2]

    # ── GPU toggle (legacy API, no-ops) ───────────────────────────
    def enable_gpu(self) -> None:
        """Legacy method — the new pipeline uses config.paddle_use_gpu."""
        logger.info("enable_gpu() called — set config.paddle_use_gpu=True instead")

    def disable_gpu(self) -> None:
        """Legacy method — the new pipeline uses config.paddle_use_gpu."""
        logger.info("disable_gpu() called — set config.paddle_use_gpu=False instead")

    # ── Classifier injection (legacy API, no-op) ──────────────────
    def set_classifier(self, classifier: Any) -> None:
        """Legacy method — classifier routing is now built-in (digital vs scanned)."""
        self._classifier = classifier
        logger.info(
            "set_classifier() called — classifier will be ignored (jb_detection "
            "auto-detects digital vs scanned PDFs natively)."
        )

    # ── Pattern configuration ─────────────────────────────────────
    def set_patterns(
        self,
        jb_examples: Optional[Union[str, List[str]]] = None,
        mc_examples: Optional[Union[str, List[str]]] = None,
        spare_examples: Optional[Union[str, List[str]]] = None,
        cable_examples: Optional[Union[str, List[str]]] = None,
        wire_color_rule: Optional[str] = None,
        scr_number_rule: Optional[str] = None,
    ) -> None:
        """Set custom patterns. Only non-None arguments are applied."""
        self._pattern_matcher.set_patterns(
            jb_examples=jb_examples,
            mc_examples=mc_examples,
            spare_examples=spare_examples,
            cable_examples=cable_examples,
            wire_color_rule=wire_color_rule,
            scr_number_rule=scr_number_rule,
        )
        self._excel_exporter.set_wire_color_rule(
            wire_color_rule if wire_color_rule is not None
            else self._pattern_matcher.wire_color_rule
        )
        self._excel_exporter.set_scr_number_rule(
            scr_number_rule if scr_number_rule is not None
            else self._pattern_matcher.scr_number_rule
        )
        # Mirror onto self for backward compatibility
        self.jb_examples = self._pattern_matcher.jb_examples
        self.mc_examples = self._pattern_matcher.mc_examples
        self.spare_examples = self._pattern_matcher.spare_examples
        self.jb_examples_list = self._pattern_matcher.jb_examples_list
        self.mc_examples_list = self._pattern_matcher.mc_examples_list
        self.spare_examples_list = self._pattern_matcher.spare_examples_list
        self.cable_examples = self._pattern_matcher.cable_examples
        self.wire_color_rule = self._pattern_matcher.wire_color_rule
        self.scr_number_rule = self._pattern_matcher.scr_number_rule
        self.jb_regex = self._pattern_matcher.jb_regex
        self.mc_regex = self._pattern_matcher.mc_regex
        self.spare_regex = self._pattern_matcher.spare_regex
        self._excel_exporter.set_spare_examples(self.spare_examples)

    def set_wire_color_rule(self, rule: Optional[str]) -> None:
        self._pattern_matcher.set_wire_color_rule(rule)
        self._excel_exporter.set_wire_color_rule(rule)
        self.wire_color_rule = rule

    def set_scr_number_rule(self, rule: Optional[str]) -> None:
        self._pattern_matcher.set_scr_number_rule(rule)
        self._excel_exporter.set_scr_number_rule(rule)
        self.scr_number_rule = rule

    def set_terminal_wire_patterns(self, config: Dict[str, Any]) -> None:
        self._pattern_matcher.set_terminal_wire_patterns(config)
        self._excel_exporter.set_terminal_wire_patterns(config)
        self.terminal_pattern = config.get("terminal_pattern", "")
        self.wire_color_rule = config.get("wire_color_pattern", "")
        self.terminal_pattern_dict = config

    # ── Tag vector building ───────────────────────────────────────
    def build_tag_vectors_from_excel(self, excel_path: str) -> None:
        """Build tag vectors from an IO List Excel file."""
        if not os.path.exists(excel_path):
            logger.error("Excel file not found: %s", excel_path)
            return
        try:
            count = self._tag_matcher.build_from_excel(excel_path)
            logger.info("Built %d tag vectors from %s", count, excel_path)
            self.tag_patterns = {
                tag: self._tag_matcher.create_tag_vector(tag)
                for tag in self._tag_matcher.reference_tags
            }
        except Exception as exc:
            logger.error("Error building tag vectors: %s", exc)
            raise

    # ── Wire / SCR / Terminal generators (delegate to exporter) ────
    def generate_scr_number(self, tag_number: int) -> str:
        return self._excel_exporter.generate_scr_number(tag_number)

    def generate_terminal_numbers(self, tag_number: int) -> Dict[str, str]:
        return self._excel_exporter.generate_terminal_numbers(tag_number)

    def generate_mc_wire_colors(self, tag_number: int) -> str:
        return self._excel_exporter.generate_mc_wire_colors(tag_number)

    def generate_mc_wire_colors_enhanced(self, tag_number: int) -> str:
        return self._excel_exporter.generate_mc_wire_colors_enhanced(tag_number)

    def extract_pair_number(self, cable_description: str) -> Optional[str]:
        return self._excel_exporter.extract_pair_number(cable_description)

    def add_wire_colors_and_scr_to_dataframe(
        self,
        df: Any,
        tag_to_number: Dict[str, int],
        output_path: str,
        pdf_results: Dict[str, Dict[int, Any]],
        io_tags: Optional[Set[str]] = None,
        pdf_name: Optional[str] = None,
    ) -> Any:
        """Build the intermediate Excel from PDF results."""
        return self._excel_exporter.create_intermediate_excel(
            page_results=pdf_results,
            output_path=output_path,
            master_tag_numbers=tag_to_number,
            io_tags=io_tags or set(),
        )

    # ── Stats ──────────────────────────────────────────────────────
    def get_processing_stats(self) -> Dict[str, Any]:
        """Return processing statistics (backward-compatible shape)."""
        total_tags = max(1, len(self.all_tags))
        matched = len(self.matched_tags)
        proc_stats = self._processor.stats()
        return {
            "total_tags": len(self.all_tags),
            "matched_tags": matched,
            "exact_matches": self.exact_matches,
            "similar_matches": self.similar_matches,
            "total_jbs": len(self.all_jbs),
            "total_mcs": len(self.all_mcs),
            "total_spares": len(self.all_spares),
            "processing_time": (
                f"{self.processing_time:.2f} seconds"
                if self.processing_time else "0.00 seconds"
            ),
            "match_rate": f"{(matched / total_tags * 100):.1f}%",
            "exact_match_rate": f"{(self.exact_matches / total_tags * 100):.0f}%",
            "unmatched_tags": len(self.all_tags) - matched,
            "pages_processed": proc_stats.get("pages_processed", 0),
            "pages_digital": proc_stats.get("pages_digital", 0),
            "pages_scanned": proc_stats.get("pages_scanned", 0),
            "total_detections": proc_stats.get("total_detections", 0),
            "pdf_types": proc_stats.get("pdf_types", {}),
        }

    def reset_stats(self) -> None:
        self.all_tags.clear()
        self.matched_tags.clear()
        self.all_jbs.clear()
        self.all_mcs.clear()
        self.all_spares.clear()
        self.exact_matches = 0
        self.similar_matches = 0
        self.processing_time = 0.0
        self._processor.reset_stats()
        self._tag_matcher.reset_stats()

    # ── Single-image extraction (returns the legacy 9-tuple) ──────
    def extract_from_image(self, image: Any) -> Tuple[Any, ...]:
        """Extract tags / JBs / MCs / cables / spares from a single image.

        Returns the legacy 9-tuple:
            (tags, jb_identifiers, mc_identifiers, cable_descriptions,
             spare_identifiers, tag_to_number, raw_cable_descriptions,
             tag_match_info, all_ocr_tags)
        """
        # Use the unified processor's _process_detections path but with
        # the detector directly on the image.
        from .image_preprocessor import preprocess
        from .models import JBDetectionResult, TagMatchInfo

        if image is None or (hasattr(image, 'size') and image.size == 0):
            return JBDetectionResult().to_tuple()

        # Preprocess
        try:
            preprocessed = preprocess(image, config=self._config)
        except Exception as exc:
            logger.warning("preprocess failed: %s — using raw image", exc)
            preprocessed = image

        # OCR (PaddleOCR)
        try:
            detections = self._processor.detector.detect(preprocessed)
        except Exception as exc:
            logger.error("OCR failed: %s", exc)
            detections = []

        # Pattern match + tag match
        result = self._processor._process_detections(detections, page_number=1)

        # Mirror onto self for backward compatibility
        self.all_tags.update(result.tags)
        self.all_jbs.update(result.jb_identifiers)
        self.all_mcs.update(result.mc_identifiers)
        self.all_spares.extend(result.spare_identifiers)
        for info in result.tag_match_info.values():
            if info.match_type == "exact":
                self.exact_matches += 1
                self.matched_tags.add(info.matched_tag)
            elif info.match_type == "similar":
                self.similar_matches += 1
                self.matched_tags.add(info.matched_tag)

        return result.to_tuple()

    # ── Single-PDF processing ─────────────────────────────────────
    def process_pdf(self, pdf_path: str) -> Dict[int, Tuple[Any, ...]]:
        """Process every page of a PDF.

        Returns Dict[int, 9-tuple] (1-indexed page numbers).
        """
        results = self._processor.process_pdf(pdf_path)
        # Convert to legacy 9-tuple form
        legacy: Dict[int, Tuple[Any, ...]] = {}
        for page_num, result in results.items():
            self.all_tags.update(result.tags)
            self.all_jbs.update(result.jb_identifiers)
            self.all_mcs.update(result.mc_identifiers)
            self.all_spares.extend(result.spare_identifiers)
            for info in result.tag_match_info.values():
                if info.match_type == "exact":
                    self.exact_matches += 1
                    self.matched_tags.add(info.matched_tag)
                elif info.match_type == "similar":
                    self.similar_matches += 1
                    self.matched_tags.add(info.matched_tag)
            legacy[page_num] = result.to_tuple()
        self.page_results = legacy
        return legacy

    def process_multiple_pdfs(
        self,
        pdf_paths: List[str],
    ) -> Dict[str, Dict[int, Tuple[Any, ...]]]:
        """Process multiple PDFs. Returns ``{pdf_name: {page: 9-tuple}}``."""
        all_results = self._processor.process_multiple_pdfs(pdf_paths)
        legacy: Dict[str, Dict[int, Tuple[Any, ...]]] = {}
        for pdf_name, page_dict in all_results.items():
            legacy[pdf_name] = {
                page_num: result.to_tuple()
                for page_num, result in page_dict.items()
            }
        return legacy

    # ── Annotated PDF ──────────────────────────────────────────────
    def create_annotated_pdf(
        self,
        pdf_path: str,
        output_pdf_path: str,
        all_pdf_results: Optional[Dict[str, Dict[int, JBDetectionResult]]] = None,
    ) -> Dict[str, int]:
        """Generate an annotated PDF with bounding boxes."""
        from .annotator import PDFAnnotator
        annotator = PDFAnnotator(config=self._config)

        # If we have results from a recent process_pdf call, use them;
        # otherwise re-run the pipeline.
        if not hasattr(self, "_last_pdf_results") or not self._last_pdf_results:
            page_results_struct = self._processor.process_pdf(pdf_path)
        else:
            page_results_struct = self._last_pdf_results

        # Stash for cross-page duplicate detection
        self._last_pdf_results = page_results_struct

        pdf_name = os.path.basename(pdf_path)
        all_results = all_pdf_results or {pdf_name: page_results_struct}

        # Build a master tag_to_number across all pages
        master_tag_to_number: Dict[str, int] = {}
        for pr in page_results_struct.values():
            master_tag_to_number.update(pr.tag_to_number)

        return annotator.annotate_pdf(
            pdf_path=pdf_path,
            page_results=page_results_struct,
            output_path=output_pdf_path,
            tag_to_number=master_tag_to_number,
            all_pdf_results=all_results,
            pdf_name=pdf_name,
        )

    # ── Excel processing (delegate to exporter) ───────────────────
    def process_excel_with_io_list(
        self,
        intermediate_excel_path: str,
        excel_path: str,
        output_path: str,
        all_ocr_tags: Optional[Set[str]] = None,
    ) -> Tuple[Any, List[str], List[str]]:
        """Merge intermediate Excel with IO List."""
        return self._excel_exporter.create_final_excel(
            intermediate_excel_path, excel_path, output_path, all_ocr_tags,
        )

    # ── Cross-page warning collection ─────────────────────────────
    def _collect_cross_page_warnings(
        self,
        all_pdf_results: Dict[str, Dict[int, JBDetectionResult]],
    ) -> List[Dict[str, Any]]:
        """Detect duplicate JBs / tags across pages and PDFs."""
        warnings_list: List[Dict[str, Any]] = []
        if not all_pdf_results:
            return warnings_list

        for pdf_name, page_dict in all_pdf_results.items():
            if not page_dict:
                continue
            jb_pages: Dict[str, List[int]] = {}
            tag_pages: Dict[str, List[int]] = {}
            for page_num, pr in page_dict.items():
                if not isinstance(pr, JBDetectionResult):
                    continue
                for jb in pr.jb_identifiers:
                    jb_u = str(jb).strip().upper()
                    if jb_u:
                        jb_pages.setdefault(jb_u, []).append(int(page_num))
                for tag in pr.tags:
                    tu = str(tag).strip().upper()
                    if tu and "SPARE" not in tu:
                        tag_pages.setdefault(tu, []).append(int(page_num))

            for jb_u, pages in jb_pages.items():
                if len(pages) > 1:
                    warnings_list.append({
                        "type": "DUPLICATE_JB",
                        "item": jb_u,
                        "pages": sorted(set(pages)),
                        "tag_count": 0,
                        "severity": "WARNING",
                        "description": (
                            f"JB '{jb_u}' appears on {len(pages)} pages of "
                            f"'{pdf_name}': {sorted(set(pages))}."
                        ),
                        "action": "Verify the JB label on each listed page.",
                        "pdf_name": pdf_name,
                    })

            for tag_u, pages in tag_pages.items():
                if len(pages) > 1:
                    warnings_list.append({
                        "type": "DUPLICATE_TAG",
                        "item": tag_u,
                        "pages": sorted(set(pages)),
                        "tag_count": 0,
                        "severity": "WARNING",
                        "description": (
                            f"Tag '{tag_u}' appears on {len(pages)} pages of "
                            f"'{pdf_name}': {sorted(set(pages))}."
                        ),
                        "action": "Verify the tag is intended to be on multiple pages.",
                        "pdf_name": pdf_name,
                    })

        # Cross-PDF duplicate tags
        cross_tag_pdfs: Dict[str, List[str]] = {}
        for pdf_name, page_dict in all_pdf_results.items():
            if not page_dict:
                continue
            pdf_tag_set: Set[str] = set()
            for pr in page_dict.values():
                if not isinstance(pr, JBDetectionResult):
                    continue
                for tag in pr.tags:
                    tu = str(tag).strip().upper()
                    if tu and "SPARE" not in tu:
                        pdf_tag_set.add(tu)
            for tu in pdf_tag_set:
                cross_tag_pdfs.setdefault(tu, []).append(pdf_name)

        for tu, pdfs in cross_tag_pdfs.items():
            if len(pdfs) > 1:
                warnings_list.append({
                    "type": "DUPLICATE_TAG",
                    "item": tu,
                    "pages": sorted(set(pdfs)),
                    "tag_count": 0,
                    "severity": "INFO",
                    "description": (
                        f"Tag '{tu}' appears in {len(pdfs)} different PDFs: "
                        f"{sorted(set(pdfs))}."
                    ),
                    "action": "Verify the tag is expected across PDFs.",
                    "pdf_name": ",".join(sorted(set(pdfs))),
                })

        return warnings_list

    # ── run() — minimal API used by DataAnalysis.run ─────────────
    def run(
        self,
        pdf_paths: List[str],
        excel_path: str,
        output_excel_path: str,
        intermediate_excel_path: str,
        all_ocr_tags: Optional[Set[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Run the pipeline and produce the final Excel.

        Returns (unmatched_io_tags, unmatched_pdf_tags).
        """
        start_time = time.time()
        try:
            self.build_tag_vectors_from_excel(excel_path)

            all_pdf_results = self.process_multiple_pdfs(pdf_paths)

            master_tag_numbers: Dict[str, int] = {}
            for page_dict in all_pdf_results.values():
                for pr_tuple in page_dict.values():
                    if isinstance(pr_tuple, tuple) and len(pr_tuple) > 5:
                        master_tag_numbers.update(pr_tuple[5] or {})

            structured_results: Dict[str, Dict[int, JBDetectionResult]] = {}
            all_ocr_tags_collected: Set[str] = set()
            for pdf_name, page_dict in all_pdf_results.items():
                structured_results[pdf_name] = {}
                for page_num, tpl in page_dict.items():
                    result = JBDetectionResult.from_tuple(tpl)
                    structured_results[pdf_name][page_num] = result
                    all_ocr_tags_collected.update(result.all_ocr_tags)

            self._excel_exporter.create_intermediate_excel(
                page_results=structured_results,
                output_path=intermediate_excel_path,
                master_tag_numbers=master_tag_numbers,
            )

            if not all_ocr_tags:
                all_ocr_tags = all_ocr_tags_collected
            _, unmatched_io, unmatched_pdf = self._excel_exporter.create_final_excel(
                intermediate_excel_path, excel_path, output_excel_path,
                all_ocr_tags,
            )

            self.processing_time = time.time() - start_time
            return unmatched_io, unmatched_pdf

        except Exception as exc:
            logger.error("run() failed: %s", exc)
            self.processing_time = time.time() - start_time
            return [], []

    # ── run_with_annotated_pdf() — full pipeline + annotated PDFs ─
    def run_with_annotated_pdf(
        self,
        pdf_paths: List[str],
        excel_path: str,
        output_excel_path: str,
        output_pdf_dir: str,
        create_zip: bool = True,
        zip_path: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """Full pipeline: PDFs → annotated PDFs + Excel + ZIP.

        Returns (io_only_tags, ocr_only_unmatched).
        """
        import shutil
        import zipfile

        start_time = time.time()
        try:
            self._page_warnings = []
            self.latest_warnings = []

            self.build_tag_vectors_from_excel(excel_path)

            io_tags: Set[str] = set()
            if excel_path and os.path.exists(excel_path):
                try:
                    import pandas as pd  # type: ignore
                    io_df = pd.read_excel(excel_path)
                    col = self._config.excel_io_list_tag_column
                    if col not in io_df.columns:
                        for alt in ("Tag", "Tag No", "Tag No.", "TAG"):
                            if alt in io_df.columns:
                                col = alt
                                break
                    if col in io_df.columns:
                        io_tags = {
                            str(t).strip().upper()
                            for t in io_df[col]
                            if pd.notna(t) and str(t).strip()
                        }
                except Exception as exc:
                    logger.error("Error reading IO List: %s", exc)

            os.makedirs(output_pdf_dir, exist_ok=True)
            all_pdf_ocr_tags: Set[str] = set()
            all_pdf_tags: Set[str] = set()
            master_tag_numbers: Dict[str, int] = {}
            all_pdf_results: Dict[str, Dict[int, JBDetectionResult]] = {}
            output_files: List[str] = []

            for pdf_path in pdf_paths:
                pdf_filename = os.path.basename(pdf_path)
                try:
                    page_results = self._processor.process_pdf(pdf_path)
                    all_pdf_results[pdf_filename] = page_results
                    self._last_pdf_results = page_results

                    for page_num, pr in page_results.items():
                        all_pdf_tags.update(pr.tags)
                        all_pdf_ocr_tags.update(pr.all_ocr_tags)
                        master_tag_numbers.update(pr.tag_to_number)

                    output_pdf_path = os.path.join(
                        output_pdf_dir, f"annotated_{pdf_filename}"
                    )
                    self.create_annotated_pdf(
                        pdf_path, output_pdf_path,
                        all_pdf_results=all_pdf_results,
                    )
                    output_files.append(output_pdf_path)
                except Exception as exc:
                    logger.error("Error processing PDF %s: %s", pdf_filename, exc)
                    continue

            intermediate_excel_path = os.path.join(
                output_pdf_dir, "JB_Wiring_Diagram_Intermediate.xlsx"
            )
            try:
                self._excel_exporter.create_intermediate_excel(
                    page_results=all_pdf_results,
                    output_path=intermediate_excel_path,
                    master_tag_numbers=master_tag_numbers,
                    io_tags=io_tags,
                )
                output_files.append(intermediate_excel_path)
            except Exception as exc:
                logger.error("Error creating intermediate Excel: %s", exc)

            unmatched_pdf_tags: List[str] = []
            unmatched_io_tags: List[str] = []
            if excel_path and os.path.exists(excel_path):
                try:
                    if not output_excel_path.endswith(".xlsx"):
                        output_excel_path = (
                            output_excel_path.replace(".xls", ".xlsx")
                            if output_excel_path.endswith(".xls")
                            else f"{output_excel_path}.xlsx"
                        )
                    if not os.path.basename(output_excel_path):
                        output_excel_path = os.path.join(
                            output_pdf_dir, "JB_Wiring_Diagram_Final.xlsx"
                        )
                    _, unmatched_io_tags, unmatched_pdf_tags = (
                        self._excel_exporter.create_final_excel(
                            intermediate_excel_path, excel_path,
                            output_excel_path, all_pdf_ocr_tags,
                        )
                    )
                    output_files.append(output_excel_path)
                except Exception as exc:
                    logger.error("Error combining with IO List: %s", exc)
                    try:
                        shutil.copy2(intermediate_excel_path, output_excel_path)
                        output_files.append(output_excel_path)
                    except Exception:
                        pass
            else:
                if not output_excel_path.endswith(".xlsx"):
                    output_excel_path = (
                        output_excel_path.replace(".xls", ".xlsx")
                        if output_excel_path.endswith(".xls")
                        else f"{output_excel_path}.xlsx"
                    )
                if not os.path.basename(output_excel_path):
                    output_excel_path = os.path.join(
                        output_pdf_dir, "JB_Wiring_Diagram_Final.xlsx"
                    )
                try:
                    shutil.copy2(intermediate_excel_path, output_excel_path)
                    output_files.append(output_excel_path)
                except Exception:
                    pass
                unmatched_pdf_tags = sorted(list(all_pdf_tags - io_tags)) if io_tags else []
                unmatched_io_tags = sorted(list(io_tags - all_pdf_tags)) if io_tags else []

            # Always create unmatched Excel
            unmatched_excel_path = os.path.join(
                output_pdf_dir, "JB_Wiring_Diagram_Unmatched_Tags.xlsx"
            )
            try:
                ocr_only_unmatched = set()
                if all_pdf_ocr_tags:
                    ocr_only_unmatched = all_pdf_ocr_tags - io_tags
                io_only_tags = io_tags - all_pdf_ocr_tags
                self._excel_exporter.create_unmatched_excel(
                    list(ocr_only_unmatched), list(io_only_tags),
                    unmatched_excel_path,
                )
                output_files.append(unmatched_excel_path)
            except Exception as exc:
                logger.error("Error creating unmatched Excel: %s", exc)

            # Collect warnings
            try:
                cross_warnings = self._collect_cross_page_warnings(all_pdf_results)
                merged = list(self._page_warnings) + list(cross_warnings)
                seen: Set[Tuple[str, str, str]] = set()
                deduped: List[Dict[str, Any]] = []
                for w in merged:
                    k = (
                        str(w.get("type", "")).upper(),
                        str(w.get("item", "")).upper(),
                        str(w.get("pdf_name", "")).upper(),
                    )
                    if k in seen:
                        continue
                    seen.add(k)
                    deduped.append(w)
                self.latest_warnings = deduped

                warnings_excel_path = os.path.join(
                    output_pdf_dir, "JB_Wiring_Diagram_Warnings.xlsx"
                )
                try:
                    self._excel_exporter.create_warnings_excel(
                        self.latest_warnings, warnings_excel_path,
                    )
                    output_files.append(warnings_excel_path)
                except Exception as exc:
                    logger.error("Error creating warnings Excel: %s", exc)
            except Exception as exc:
                logger.error("Error collecting warnings: %s", exc)
                self.latest_warnings = list(self._page_warnings)

            if create_zip:
                try:
                    if zip_path is None:
                        zip_path = output_pdf_dir.rstrip("/\\") + ".zip"
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                        for fp in output_files:
                            if os.path.exists(fp):
                                zf.write(fp, arcname=os.path.basename(fp))
                    logger.info("ZIP archive created: %s", zip_path)
                except Exception as exc:
                    logger.error("Error creating ZIP: %s", exc)

            self.processing_time = time.time() - start_time
            logger.info(
                "PROCESSING COMPLETED: %d PDFs, %d OCR tags, %d IO tags, "
                "unmatched_pdf=%d, unmatched_io=%d, time=%.2fs",
                len(pdf_paths), len(all_pdf_ocr_tags), len(io_tags),
                len(unmatched_pdf_tags), len(unmatched_io_tags),
                self.processing_time,
            )

            io_only_tags_final = io_tags - all_pdf_ocr_tags
            ocr_only_unmatched_final = all_pdf_ocr_tags - io_tags
            return (list(io_only_tags_final), list(ocr_only_unmatched_final))

        except Exception as exc:
            logger.error("CRITICAL ERROR in run_with_annotated_pdf: %s", exc)
            self.processing_time = time.time() - start_time
            self.latest_warnings = list(self._page_warnings)
            return [], []

    # ── Legacy _create_unmatched_tags_excel ─────────────────────────
    def _create_unmatched_tags_excel(
        self,
        unmatched_excel_tags: List[str],
        unmatched_pdf_tags: List[str],
        output_path: str,
    ) -> None:
        """Create an Excel file listing unmatched tags.

        Legacy signature: (unmatched_excel_tags, unmatched_pdf_tags, output_path)
        where unmatched_excel_tags = IO List tags NOT in PDFs, and
        unmatched_pdf_tags = PDF OCR tags NOT in IO List.
        """
        try:
            self._excel_exporter.create_unmatched_excel(
                ocr_only_tags=unmatched_pdf_tags,
                io_only_tags=unmatched_excel_tags,
                output_path=output_path,
            )
            logger.info("Unmatched tags Excel saved: %s", output_path)
        except Exception as exc:
            logger.error("Error creating unmatched Excel: %s", exc)

    # ── Legacy _create_empty_excel (compat no-op) ──────────────────
    def _create_empty_excel(self, file_path: str) -> None:
        """Legacy method — create an empty Excel file."""
        try:
            import openpyxl
            wb = openpyxl.Workbook()
            wb.save(file_path)
        except Exception as exc:
            logger.error("Error creating empty Excel: %s", exc)

    # ── Legacy _create_report_file ─────────────────────────────────
    def _create_report_file(
        self,
        reports_path: str,
        unmatched_excel_tags: List[str],
        unmatched_pdf_tags: List[str],
    ) -> None:
        """Legacy method — create a JSON report file."""
        import json
        report = {
            "processing_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "unmatched_excel_tags_count": len(unmatched_excel_tags),
            "unmatched_pdf_tags_count": len(unmatched_pdf_tags),
            "unmatched_excel_tags": unmatched_excel_tags[:100],
            "unmatched_pdf_tags": unmatched_pdf_tags[:100],
        }
        try:
            with open(reports_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error("Error creating report file: %s", exc)

    # ── Legacy _create_zip_archive ─────────────────────────────────
    def _create_zip_archive(
        self,
        zip_path: str,
        files_to_add: List[str],
    ) -> None:
        """Legacy method — create a ZIP archive."""
        import zipfile
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for fp in files_to_add:
                    if os.path.exists(fp):
                        zf.write(fp, arcname=os.path.basename(fp))
        except Exception as exc:
            logger.error("Error creating ZIP: %s", exc)

    # ── Legacy check_tag_number_consistency ────────────────────────
    def check_tag_number_consistency(
        self,
        tag_to_number: Dict[str, int],
    ) -> Tuple[bool, int, int]:
        """Legacy method — check if tag numbers are consistent."""
        if not tag_to_number:
            return (True, 0, 0)
        numbers = list(tag_to_number.values())
        expected = list(range(1, len(numbers) + 1))
        is_consistent = sorted(numbers) == expected
        return (is_consistent, min(numbers), max(numbers))


# ═══════════════════════════════════════════════════════════════════════════
# Compat DataAnalysis (wraps TagJBExtractor + optional classifier)
# ═══════════════════════════════════════════════════════════════════════════
class DataAnalysis:
    """Drop-in replacement for ``DataAnalysisModule.DataAnalysis``.

    Same constructor signature as the legacy class. The PDF classifier
    is OPTIONAL — if ``pdf_classifier`` isn't installed or the model
    files are missing, ``self.classifier`` is ``None`` and document type
    detection is handled natively by the unified processor (digital vs
    scanned auto-detection).
    """

    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "modules", "keras_model.h5",
    )
    DEFAULT_LABELS_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "modules", "labels.txt",
    )
    DEFAULT_PDF_TYPE = "diagrams"

    def __init__(
        self,
        extractor: Optional[TagJBExtractor] = None,
        classifier_model_path: Optional[str] = None,
        classifier_labels_path: Optional[str] = None,
    ) -> None:
        # If no extractor is provided, create one (legacy behavior).
        if extractor is None:
            extractor = TagJBExtractor()
        self.extractor = extractor
        self.classifier = None
        self.document_types: Dict[str, str] = {}
        self.classifier_model_path = classifier_model_path or self.DEFAULT_MODEL_PATH
        self.classifier_labels_path = classifier_labels_path or self.DEFAULT_LABELS_PATH
        self._load_classifier()

    def _load_classifier(self) -> None:
        """Try to load the optional Keras PDF classifier.

        If unavailable, document type detection falls back to the
        unified processor's native digital/scanned detection.
        """
        try:
            from pdf_classifier import PDFClassifier  # type: ignore
        except ImportError:
            try:
                from PDFClassifier import PDFClassifier  # type: ignore
            except ImportError:
                logger.info(
                    "PDFClassifier backend unavailable; document type "
                    "detection will use native digital/scanned auto-detection."
                )
                return

        if (not os.path.exists(self.classifier_model_path)
                or not os.path.exists(self.classifier_labels_path)):
            logger.warning(
                "PDFClassifier assets missing: %s, %s — using native detection",
                self.classifier_model_path, self.classifier_labels_path,
            )
            return

        try:
            self.classifier = PDFClassifier(
                model_path=self.classifier_model_path,
                labels_path=self.classifier_labels_path,
            )
            if hasattr(self.extractor, "set_classifier"):
                self.extractor.set_classifier(self.classifier)
            logger.info(
                "DataAnalysis initialized with PDFClassifier: %s",
                self.classifier_model_path,
            )
        except Exception as exc:
            logger.error("Failed to initialize PDFClassifier: %s", exc)
            self.classifier = None

    def detect_pdf_type(self, pdf_path: str) -> str:
        """Detect PDF type (returns 'diagrams' or 'table').

        If the classifier is unavailable, returns the default type.
        """
        if self.classifier is None:
            return self.DEFAULT_PDF_TYPE
        try:
            raw_label = self.classifier.classify_pdf(pdf_path)
            label_l = (raw_label or "").strip().lower()
            if "table" in label_l:
                pdf_type = "table"
            elif "diagram" in label_l or "drawing" in label_l:
                pdf_type = "diagrams"
            else:
                pdf_type = self.DEFAULT_PDF_TYPE
            logger.info(
                "PDFClassifier returned '%s' → normalized to '%s'",
                raw_label, pdf_type,
            )
            self.document_types[pdf_path] = pdf_type
            return pdf_type
        except Exception as exc:
            logger.warning(
                "PDFClassifier failed for %s: %s — defaulting to %s",
                os.path.basename(pdf_path), exc, self.DEFAULT_PDF_TYPE,
            )
            self.document_types[pdf_path] = self.DEFAULT_PDF_TYPE
            return self.DEFAULT_PDF_TYPE

    def run_with_annotated_pdf(
        self,
        pdf_paths: List[str],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[List[str], List[str]]:
        """Delegate to the underlying extractor."""
        if self.classifier is not None and isinstance(pdf_paths, list):
            for pdf_path in pdf_paths:
                self.document_types[pdf_path] = self.detect_pdf_type(pdf_path)
                if hasattr(self.extractor, "document_type_by_path"):
                    self.extractor.document_type_by_path[pdf_path] = (
                        self.document_types[pdf_path]
                    )
        return self.extractor.run_with_annotated_pdf(pdf_paths, *args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Tuple[List[str], List[str]]:
        """Delegate to the underlying extractor."""
        return self.extractor.run(*args, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        # Delegate any unknown attribute access to the underlying extractor.
        # __getattr__ is only called when normal lookup fails, so
        # self.extractor itself is resolved via the normal mechanism.
        return getattr(self.extractor, attr)


__all__ = ["TagJBExtractor", "DataAnalysis"]
