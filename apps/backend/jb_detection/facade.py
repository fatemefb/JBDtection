"""JBDetection — public API facade.

This module exposes the two classes that the rest of the world imports:

- :class:`TagJBExtractor` — the workhorse. Wraps the
  :class:`JBDetectionPipeline`, :class:`TagMatcher`, and
  :class:`ExcelExporter` into a single object whose public method
  signatures are **byte-for-byte compatible** with the original
  11,000-line monolith.
- :class:`DataAnalysis` — a thin facade on top of
  :class:`TagJBExtractor` that adds optional PDF type detection via
  a Keras-based classifier. Same constructor signature, same
  ``__getattr__`` delegation pattern.

Backward compatibility notes
----------------------------
- ``tesseract_path`` is accepted by :meth:`TagJBExtractor.__init__`
  but **ignored** (we use PaddleOCR now). It is kept so existing
  callers don't break.
- ``pdf_classifier`` import is wrapped in try/except — same as the
  original. If the classifier module is unavailable,
  ``DataAnalysis.classifier`` is ``None`` and detection defaults to
  ``"diagrams"``.
- The 9-tuple extraction output is preserved via
  :meth:`TagJBExtractor.extract_from_image` and
  :meth:`TagJBExtractor.process_pdf` (returns a dict of 9-tuples).
- The 16-column Excel output is produced by :class:`ExcelExporter`
  with the EXACT column order.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .config import Config, load_config
from .excel_exporter import ExcelExporter
from .models import JBDetectionResult, PageResult, TagMatchInfo
from .pattern_matcher import PatternMatcher
from .pipeline import JBDetectionPipeline
from .tag_matcher import TagMatcher

logger = logging.getLogger("jb_detection.facade")

current_dir = os.path.dirname(os.path.abspath(__file__))


# ── Optional PDF classifier import ──────────────────────────────────────
# Same try/except chain as the original — keeps the facade working
# even when the classifier backend isn't installed.
_PDFClassifierClass = None
try:
    from pdf_classifier import PDFClassifier as _PDFClassifierClass  # type: ignore
except ImportError:
    try:
        from PDFClassifier import PDFClassifier as _PDFClassifierClass  # type: ignore
    except ImportError:
        _PDFClassifierClass = None


# ════════════════════════════════════════════════════════════════════════
# TagJBExtractor
# ════════════════════════════════════════════════════════════════════════
class TagJBExtractor:
    """Public-facing facade around the JBDetection pipeline.

    Constructor signature is identical to the original
    ``TagJBExtractor``. ``tesseract_path`` is accepted but ignored
    (we use PaddleOCR now).
    """

    def __init__(self,
                 tesseract_path: Optional[str] = None,
                 excel_path: Optional[str] = None) -> None:
        # tesseract_path is IGNORED — kept for backward compatibility.
        # We do NOT raise if it doesn't exist (the original did, but
        # since we no longer use tesseract there's no reason to).
        if tesseract_path:
            logger.info("tesseract_path=%s ignored (JBDetection uses PaddleOCR)",
                         tesseract_path)

        # ── Configuration ─────────────────────────────────────────
        self._config: Config = load_config()

        # ── Collaborators ─────────────────────────────────────────
        self._pattern_matcher = PatternMatcher()
        self._tag_matcher = TagMatcher(config=self._config)
        self._excel_exporter = ExcelExporter(config=self._config)
        self._excel_exporter.set_pattern_matcher(self._pattern_matcher)
        self._pipeline = JBDetectionPipeline(
            config=self._config,
            pattern_matcher=self._pattern_matcher,
            tag_matcher=self._tag_matcher,
        )

        # ── State (preserved from the original) ───────────────────
        self.excel_path = excel_path
        self.all_tags: Set[str] = set()
        self.matched_tags: Set[str] = set()
        self.all_jbs: Set[str] = set()
        self.all_mcs: Set[str] = set()
        self.all_spares: List[str] = []
        self.exact_matches = 0
        self.similar_matches = 0
        self.processing_time = 0
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
        self.document_type_by_path: Dict[str, str] = {}
        self.document_nature_by_path: Dict[str, str] = {}
        self.page_results: Dict[int, Any] = {}

        # If an excel_path was supplied, eagerly build the tag matcher.
        if excel_path and os.path.exists(excel_path):
            try:
                self.build_tag_vectors_from_excel(excel_path)
            except Exception as exc:
                logger.warning("Failed to build tag vectors from %s: %s",
                                excel_path, exc)

    # ── Classifier injection (backward-compatible) ────────────────
    def set_classifier(self, classifier: Any) -> None:
        """Inject a PDFClassifier instance (optional)."""
        self._classifier = classifier
        logger.info("PDFClassifier injected: %s",
                     type(classifier).__name__ if classifier is not None else "None")

    # ── Pattern configuration ─────────────────────────────────────
    def set_patterns(self,
                     jb_examples: Optional[Union[str, List[str]]] = None,
                     mc_examples: Optional[Union[str, List[str]]] = None,
                     spare_examples: Optional[Union[str, List[str]]] = None,
                     cable_examples: Optional[Union[str, List[str]]] = None,
                     wire_color_rule: Optional[str] = None,
                     scr_number_rule: Optional[str] = None) -> None:
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

    # ── Tag vector building ────────────────────────────────────────
    def build_tag_vectors_from_excel(self, excel_path: str) -> None:
        """Build tag vectors from an IO List Excel file."""
        if not os.path.exists(excel_path):
            logger.error("Excel file not found: %s", excel_path)
            return
        try:
            count = self._tag_matcher.build_from_excel(excel_path)
            logger.info("Built %d tag vectors from %s", count, excel_path)
            # Build tag_patterns (used by some downstream code)
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

    # ── Stats ──────────────────────────────────────────────────────
    def get_processing_stats(self) -> Dict[str, Any]:
        """Return processing statistics (backward-compatible shape)."""
        total_tags = max(1, len(self.all_tags))
        matched = len(self.matched_tags)
        return {
            "total_tags": len(self.all_tags),
            "matched_tags": matched,
            "exact_matches": self.exact_matches,
            "similar_matches": self.similar_matches,
            "total_jbs": len(self.all_jbs),
            "processing_time": (
                f"{self.processing_time:.2f} seconds" if self.processing_time else "0.00 seconds"
            ),
            "match_rate": f"{(matched / total_tags * 100):.1f}%",
            "exact_match_rate": f"{(self.exact_matches / total_tags * 100):.0f}%",
            "unmatched_tags": len(self.all_tags) - matched,
        }

    def reset_stats(self) -> None:
        self.all_tags.clear()
        self.matched_tags.clear()
        self.all_jbs.clear()
        self.all_mcs.clear()
        self.all_spares.clear()
        self.exact_matches = 0
        self.similar_matches = 0
        self.processing_time = 0
        self._pipeline.reset_stats()
        self._tag_matcher.reset_stats()

    # ── Backward-compatible pattern-matcher helpers ───────────────
    def _select_best_mc_identifier(self,
                                     mc_identifiers: Union[Set[str], List[str]],
                                     jb_identifiers: Union[Set[str], List[str]]) -> str:
        return self._pattern_matcher.select_best_mc_identifier(
            mc_identifiers, jb_identifiers,
        )

    def _select_best_cable_description(self, cable_descriptions: List[str]) -> str:
        return self._pattern_matcher.select_best_cable_description(cable_descriptions)

    # ── Single-image extraction (returns the legacy 9-tuple) ──────
    def extract_from_image(self, image: Any) -> Tuple[Any, ...]:
        """Extract tags / JBs / MCs / cables / spares from a single image.

        Returns the legacy 9-tuple:
            (tags, jb_identifiers, mc_identifiers, cable_descriptions,
             spare_identifiers, tag_to_number, raw_cable_descriptions,
             tag_match_info, all_ocr_tags)
        """
        result = self._pipeline.process_page(image, page_number=1)
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

        Returns
        -------
        Dict[int, 9-tuple]
            ``{page_number: (tags, jbs, mcs, cables, spares, t2n,
                              raw_cables, tmi, ocr_tags)}``.
        """
        results = self._pipeline.process_pdf(pdf_path)
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

    def process_multiple_pdfs(self,
                                pdf_paths: List[str],
                                ) -> Dict[str, Dict[int, Tuple[Any, ...]]]:
        """Process multiple PDFs. Returns ``{pdf_name: {page: 9-tuple}}``."""
        all_results = self._pipeline.process_multiple_pdfs(pdf_paths)
        legacy: Dict[str, Dict[int, Tuple[Any, ...]]] = {}
        for pdf_name, page_dict in all_results.items():
            legacy[pdf_name] = {
                page_num: result.to_tuple()
                for page_num, result in page_dict.items()
            }
        return legacy

    # ── Annotated PDF ──────────────────────────────────────────────
    def create_annotated_pdf(self,
                              pdf_path: str,
                              output_pdf_path: str,
                              all_pdf_results: Optional[Dict[str, Dict[int, JBDetectionResult]]] = None,
                              ) -> Dict[str, int]:
        """Generate an annotated PDF with bounding boxes."""
        from .annotator import PDFAnnotator
        annotator = PDFAnnotator(config=self._config)

        # Re-run the pipeline to get per-page results (the original
        # did the same — see ``create_annotated_pdf`` in the monolith).
        # If the caller already ran process_pdf, we use that.
        if not hasattr(self, "_last_pdf_results") or not self._last_pdf_results:
            page_results_struct = self._pipeline.process_pdf(pdf_path)
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

            # Duplicate JBs within one PDF
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

            # Duplicate tags within one PDF
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
    def run(self,
            pdf_paths: List[str],
            excel_path: str,
            output_excel_path: str,
            intermediate_excel_path: str,
            all_ocr_tags: Optional[Set[str]] = None,
            ) -> Tuple[List[str], List[str]]:
        """Run the pipeline and produce the final Excel.

        Returns
        -------
        (unmatched_io_tags, unmatched_pdf_tags)
        """
        start_time = time.time()
        try:
            # 1. Build tag vectors from IO List
            self.build_tag_vectors_from_excel(excel_path)

            # 2. Process all PDFs
            all_pdf_results = self.process_multiple_pdfs(pdf_paths)

            # 3. Build intermediate Excel
            master_tag_numbers: Dict[str, int] = {}
            for page_dict in all_pdf_results.values():
                for pr_tuple in page_dict.values():
                    if isinstance(pr_tuple, tuple) and len(pr_tuple) > 5:
                        master_tag_numbers.update(pr_tuple[5] or {})

            # Convert legacy tuples to JBDetectionResult for the exporter
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

            # 4. Merge with IO List
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
    def run_with_annotated_pdf(self,
                                pdf_paths: List[str],
                                excel_path: str,
                                output_excel_path: str,
                                output_pdf_dir: str,
                                create_zip: bool = True,
                                zip_path: Optional[str] = None,
                                ) -> Tuple[List[str], List[str]]:
        """Full pipeline: PDFs → annotated PDFs + Excel + ZIP.

        Returns
        -------
        (io_only_tags, ocr_only_unmatched)
            ``io_only_tags``: IO List tags NOT found in any PDF.
            ``ocr_only_unmatched``: PDF OCR tags NOT in IO List.
        """
        start_time = time.time()
        try:
            # Reset per-run state
            self._page_warnings = []
            self.latest_warnings = []

            # 1. Build tag vectors
            self.build_tag_vectors_from_excel(excel_path)

            # 2. Load IO tags
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

            # 3. Process all PDFs
            os.makedirs(output_pdf_dir, exist_ok=True)
            all_pdf_ocr_tags: Set[str] = set()
            all_pdf_tags: Set[str] = set()
            master_tag_numbers: Dict[str, int] = {}
            all_pdf_results: Dict[str, Dict[int, JBDetectionResult]] = {}
            output_files: List[str] = []

            for pdf_path in pdf_paths:
                pdf_filename = os.path.basename(pdf_path)
                try:
                    page_results = self._pipeline.process_pdf(pdf_path)
                    all_pdf_results[pdf_filename] = page_results
                    self._last_pdf_results = page_results

                    for page_num, pr in page_results.items():
                        all_pdf_tags.update(pr.tags)
                        all_pdf_ocr_tags.update(pr.all_ocr_tags)
                        master_tag_numbers.update(pr.tag_to_number)

                    # Create annotated PDF
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

            # 4. Create intermediate Excel
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

            # 5. Create final Excel
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
                # No IO List — copy intermediate as final
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

            # 6. Create unmatched Excel (always)
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

            # 7. Collect warnings
            try:
                cross_warnings = self._collect_cross_page_warnings(all_pdf_results)
                merged = list(self._page_warnings) + list(cross_warnings)
                # Deduplicate
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

            # 8. Create ZIP (optional)
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

            # 9. Final summary
            self.processing_time = time.time() - start_time
            logger.info(
                "PROCESSING COMPLETED: %d PDFs, %d OCR tags, %d IO tags, "
                "unmatched_pdf=%d, unmatched_io=%d, time=%.2fs",
                len(pdf_paths), len(all_pdf_ocr_tags), len(io_tags),
                len(unmatched_pdf_tags), len(unmatched_io_tags),
                self.processing_time,
            )

            # Return value matches the original:
            #   return (list(io_only_tags), list(ocr_only_unmatched))
            io_only_tags_final = io_tags - all_pdf_ocr_tags
            ocr_only_unmatched_final = all_pdf_ocr_tags - io_tags
            return (list(io_only_tags_final), list(ocr_only_unmatched_final))

        except Exception as exc:
            logger.error("CRITICAL ERROR in run_with_annotated_pdf: %s", exc)
            self.processing_time = time.time() - start_time
            self.latest_warnings = list(self._page_warnings)
            return [], []


# ════════════════════════════════════════════════════════════════════════
# DataAnalysis
# ════════════════════════════════════════════════════════════════════════
class DataAnalysis:
    """Facade on top of :class:`TagJBExtractor` with optional classifier.

    Same constructor signature as the original. The classifier is
    OPTIONAL — if ``pdf_classifier`` isn't installed or the model files
    are missing, ``self.classifier`` is ``None`` and document type
    detection defaults to ``"diagrams"``.
    """

    DEFAULT_MODEL_PATH = os.path.join(current_dir, "modules", "keras_model.h5")
    DEFAULT_LABELS_PATH = os.path.join(current_dir, "modules", "labels.txt")
    DEFAULT_PDF_TYPE = "diagrams"

    def __init__(self,
                 extractor: TagJBExtractor,
                 classifier_model_path: Optional[str] = None,
                 classifier_labels_path: Optional[str] = None) -> None:
        self.extractor = extractor
        self.classifier = None
        self.document_types: Dict[str, str] = {}
        self.classifier_model_path = classifier_model_path or self.DEFAULT_MODEL_PATH
        self.classifier_labels_path = classifier_labels_path or self.DEFAULT_LABELS_PATH
        self._load_classifier()

    def _load_classifier(self) -> None:
        if _PDFClassifierClass is None:
            logger.warning(
                "PDFClassifier backend unavailable; document type detection disabled."
            )
            return
        if (not os.path.exists(self.classifier_model_path)
                or not os.path.exists(self.classifier_labels_path)):
            logger.warning(
                "PDFClassifier assets missing; document type detection disabled: %s, %s",
                self.classifier_model_path, self.classifier_labels_path,
            )
            return
        try:
            self.classifier = _PDFClassifierClass(
                model_path=self.classifier_model_path,
                labels_path=self.classifier_labels_path,
            )
            if hasattr(self.extractor, "set_classifier"):
                self.extractor.set_classifier(self.classifier)
            logger.info("DataAnalysis initialized with PDFClassifier: %s",
                         self.classifier_model_path)
        except Exception as exc:
            logger.error("Failed to initialize PDFClassifier: %s", exc)
            self.classifier = None

    def detect_pdf_type(self, pdf_path: str) -> str:
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
            logger.info("PDFClassifier returned '%s' → normalized to '%s'",
                         raw_label, pdf_type)
            self.document_types[pdf_path] = pdf_type
            return pdf_type
        except Exception as exc:
            logger.warning("PDFClassifier failed for %s: %s — defaulting to %s",
                            os.path.basename(pdf_path), exc,
                            self.DEFAULT_PDF_TYPE)
            self.document_types[pdf_path] = self.DEFAULT_PDF_TYPE
            return self.DEFAULT_PDF_TYPE

    def run_with_annotated_pdf(self, pdf_paths: List[str], *args: Any,
                                  **kwargs: Any) -> Tuple[List[str], List[str]]:
        if self.classifier is not None and isinstance(pdf_paths, list):
            for pdf_path in pdf_paths:
                self.document_types[pdf_path] = self.detect_pdf_type(pdf_path)
                if hasattr(self.extractor, "document_type_by_path"):
                    self.extractor.document_type_by_path[pdf_path] = (
                        self.document_types[pdf_path]
                    )
        return self.extractor.run_with_annotated_pdf(pdf_paths, *args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Tuple[List[str], List[str]]:
        return self.extractor.run(*args, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        # Delegate any unknown attribute access to the underlying extractor.
        # ``__getattr__`` is only called when normal lookup fails, so
        # ``self.extractor`` itself is resolved via the normal mechanism.
        return getattr(self.extractor, attr)


__all__ = ["TagJBExtractor", "DataAnalysis"]
