"""JBDetection — Excel exporter.

Generates the four Excel artefacts produced by the pipeline:

1. **Intermediate Excel** — 16-column format with wire colors, terminals,
   SCR numbers, cable codes, per-page warnings.
2. **Final Excel** — IO List enriched with the intermediate data
   (left-join: every IO List row stays; matching PDF columns are
   filled in).
3. **Unmatched Excel** — PDF tags not in IO List, plus IO List tags
   not found in PDF.
4. **Warnings Excel** — JB-not-found, multiple-JB, duplicate-JB,
   duplicate-tag warnings collected during the run.

The exact column order is preserved from the original code so existing
downstream consumers (UI dashboards, BI pipelines) keep working:

    PDF_Name, Page, JB, MC, Tag/SPARE, Tag_Number,
    Wire_Code_1, Wire_Code_2,
    Terminal_First_Number, Terminal_Second_Number,
    Cable_Code, SCR_Terminal_Number,
    Cable_Description, Type, Tag_Number_Status, Warning
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .config import Config
from .models import JBDetectionResult, PageResult, TagMatchInfo

logger = logging.getLogger("jb_detection.excel_exporter")


# ── Column order (MUST match the original exactly) ──────────────────────
INTERMEDIATE_COLUMNS: List[str] = [
    "PDF_Name", "Page", "JB", "MC", "Tag/SPARE", "Tag_Number",
    "Wire_Code_1", "Wire_Code_2",
    "Terminal_First_Number", "Terminal_Second_Number",
    "Cable_Code", "SCR_Terminal_Number",
    "Cable_Description", "Type", "Tag_Number_Status", "Warning",
]

UNMATCHED_COLUMNS: List[str] = [
    "Tag", "Source", "Status", "Severity", "Action", "Match_Type",
]

WARNINGS_COLUMNS: List[str] = [
    "Warning_Type", "Item", "PDF_Name", "Pages",
    "Tag_Count", "Severity", "Description", "Action",
]


# ── ExcelExporter ───────────────────────────────────────────────────────
class ExcelExporter:
    """Generate the four Excel artefacts.

    The exporter accepts either structured :class:`PageResult` objects
    or the legacy 9-tuple format (for backward compatibility with the
    facade, which still works in tuples).

    Wire colors, terminal numbers, and SCR numbers are produced by
    *rules* (string templates with ``{number}`` placeholders). These
    rules are set on the exporter itself so the facade can pass them
    through.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        if config is None:
            from .config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG
        self._config = config

        # Rules — populated by facade.set_wire_color_rule / .set_scr_number_rule
        self.wire_color_rule: Optional[str] = None
        self.scr_number_rule: Optional[str] = None
        self.terminal_pattern: Optional[str] = None
        self.terminal_pattern_dict: Dict[str, Any] = {}
        self.spare_examples: Optional[str] = None  # used to build SPARE ids

    # ── Rule setters ───────────────────────────────────────────────
    def set_wire_color_rule(self, rule: Optional[str]) -> None:
        self.wire_color_rule = rule

    def set_scr_number_rule(self, rule: Optional[str]) -> None:
        self.scr_number_rule = rule

    def set_terminal_wire_patterns(self, config: Dict[str, Any]) -> None:
        self.terminal_pattern = config.get("terminal_pattern", "")
        self.wire_color_rule = config.get("wire_color_pattern", "")
        self.terminal_pattern_dict = config

    def set_spare_examples(self, spare_examples: Optional[str]) -> None:
        self.spare_examples = spare_examples

    # ── Rule evaluators (kept identical to the original behaviour) ─
    def generate_scr_number(self, tag_number: int) -> str:
        """Render the SCR terminal string for a tag number.

        Supports both ``{number}`` / ``{number:02d}`` substitutions and
        arbitrary ``{expr}`` Python expressions where ``number`` is the
        tag number. Empty rule → empty string.
        """
        if not self.scr_number_rule:
            return ""
        rule = self.scr_number_rule
        try:
            # Format spec: {number:02d}
            fmt_match = re.search(r"\{number:([^}]+)\}", rule)
            if fmt_match:
                fmt = fmt_match.group(1)
                formatted = format(tag_number, fmt)
                return rule.replace(fmt_match.group(0), formatted)
            # Plain {number}
            if "{number}" in rule:
                return rule.replace("{number}", str(tag_number))
            # Arbitrary expression {expr}
            def _repl(m: re.Match) -> str:
                expr = m.group(1).replace("number", str(tag_number))
                try:
                    # NOTE: eval is restricted to arithmetic on a single int.
                    # We do NOT expose locals/globals — only the builtins
                    # needed for math (which is empty here, so this is
                    # effectively a tiny arithmetic evaluator).
                    return str(eval(expr, {"__builtins__": {}}, {}))
                except Exception:
                    return m.group(0)
            return re.sub(r"\{([^}]+)\}", _repl, rule)
        except Exception:
            return ""

    def generate_mc_wire_colors(self, tag_number: int) -> str:
        """Render wire colors for a tag number.

        Default (no rule): ``BK{NN}, WT{NN}`` where ``NN`` is the tag
        number zero-padded to 2 digits.
        """
        if not self.wire_color_rule:
            return f"BK{tag_number:02d}, WT{tag_number:02d}"
        try:
            parts: List[str] = []
            for rule in str(self.wire_color_rule).split(","):
                rule = rule.strip()
                if not rule:
                    continue
                # {number:02d} or {number}
                fmt_match = re.search(r"\{number:([^}]+)\}", rule)
                if fmt_match:
                    fmt = fmt_match.group(1)
                    color = rule.replace(fmt_match.group(0),
                                          format(tag_number, fmt))
                elif "{number}" in rule:
                    color = rule.replace("{number}", str(tag_number))
                else:
                    # Try arbitrary expression
                    expr_match = re.search(r"\{([^}]+)\}", rule)
                    if expr_match:
                        expr = expr_match.group(1).replace(
                            "number", str(tag_number)
                        )
                        try:
                            result = eval(expr, {"__builtins__": {}}, {})
                            color = rule.replace(expr_match.group(0), str(result))
                        except Exception:
                            color = rule
                    else:
                        color = rule
                parts.append(color)
            return ", ".join(parts)
        except Exception:
            return f"BK{tag_number:02d}, WT{tag_number:02d}"

    def generate_mc_wire_colors_enhanced(self, tag_number: int) -> str:
        """Enhanced version that respects ``terminal_pattern_dict``."""
        if self.terminal_pattern_dict:
            wire_pattern = self.terminal_pattern_dict.get("wire_color_pattern", "")
            if wire_pattern:
                try:
                    def _repl(m: re.Match) -> str:
                        fmt_spec = m.group(1) or ""
                        if ":" in fmt_spec:
                            width = int(fmt_spec.split(":")[1].rstrip("d}") or 2)
                            return str(tag_number).zfill(width)
                        return str(tag_number)
                    return re.sub(r"\{x(?::(\d+)d)?\}", _repl, wire_pattern)
                except Exception:
                    pass
        return self.generate_mc_wire_colors(tag_number)

    def generate_terminal_numbers(self, tag_number: int) -> Dict[str, str]:
        """Render terminal numbers for a tag number."""
        if not self.terminal_pattern:
            return {
                "terminal_first": str(tag_number),
                "terminal_second": str(tag_number + 1),
                "scr_terminal": self.generate_scr_number(tag_number),
                "full_string": f"{tag_number}, {tag_number + 1}",
            }
        try:
            pattern = self.terminal_pattern
            include_scr = self.terminal_pattern_dict.get("include_scr", True)

            def _repl(m: re.Match) -> str:
                expr = m.group(1).replace("x", str(tag_number))
                try:
                    return str(int(eval(expr, {"__builtins__": {}}, {})))
                except Exception:
                    return m.group(0)

            result = re.sub(r"\{([^}]+)\}", _repl, pattern)
            if not include_scr:
                result = re.sub(r",?\s*SCR\s*,?", "", result)
                result = re.sub(r",\s*,", ",", result).strip(", ")

            parts = [p.strip() for p in result.split(",")]
            terminal_first = ""
            terminal_second = ""
            scr_terminal = ""
            scr_parts = [p for p in parts if "SCR" in p.upper()]
            if scr_parts:
                scr_terminal = scr_parts[0]
                parts = [p for p in parts if "SCR" not in p.upper()]
            if len(parts) >= 1:
                terminal_first = parts[0]
            if len(parts) >= 2:
                terminal_second = parts[1]
            return {
                "terminal_first": terminal_first,
                "terminal_second": terminal_second,
                "scr_terminal": scr_terminal,
                "full_string": result,
            }
        except Exception:
            return {
                "terminal_first": str(tag_number),
                "terminal_second": str(tag_number + 1),
                "scr_terminal": "",
                "full_string": f"{tag_number}, {tag_number + 1}",
            }

    # ── Best-MC / Best-Cable (delegated to pattern_matcher when present) ─
    def _select_best_cable_description(self, cable_descriptions: List[str]) -> str:
        """Pick cable code with the largest digit sum."""
        if not cable_descriptions:
            return ""
        if len(cable_descriptions) == 1:
            return str(cable_descriptions[0])

        def score(cable: str) -> Tuple[int, int, int]:
            try:
                s = str(cable).upper().strip()
                digit_groups = re.findall(r"\d+", s)
                if not digit_groups:
                    return (0, 0, 0)
                digits_int = [int(d) for d in digit_groups]
                return (sum(digits_int), max(digits_int), len(s))
            except Exception:
                return (0, 0, 0)

        seen: Set[str] = set()
        unique: List[str] = []
        for c in cable_descriptions:
            cs = str(c).strip().upper()
            if cs and cs not in seen:
                seen.add(cs)
                unique.append(c)
        if not unique:
            return ""
        if len(unique) == 1:
            return str(unique[0])
        return str(max(unique, key=score))

    # ── Intermediate Excel ─────────────────────────────────────────
    def create_intermediate_excel(
        self,
        page_results: Dict[str, Dict[int, Any]],
        output_path: str,
        master_tag_numbers: Optional[Dict[str, int]] = None,
        io_tags: Optional[Set[str]] = None,
    ) -> "Any":
        """Create the 16-column intermediate Excel.

        Parameters
        ----------
        page_results:
            ``{pdf_name: {page_number: JBDetectionResult | 9-tuple}}``.
        output_path:
            Where to save the .xlsx file.
        master_tag_numbers:
            Cross-PDF tag → number map (used as fallback when a page's
            own ``tag_to_number`` doesn't have an entry).
        io_tags:
            Set of IO List tags (uppercase). Used to mark
            ``Tag_Number_Status`` for unmatched tags.
        """
        import pandas as pd  # type: ignore

        master_tag_numbers = master_tag_numbers or {}
        io_tags = io_tags or set()
        new_df_data: List[Dict[str, Any]] = []
        page_warnings: List[Dict[str, Any]] = []

        for pdf_name, page_dict in page_results.items():
            if not page_dict:
                continue
            for page_num, page_data in page_dict.items():
                # Normalise input to JBDetectionResult
                if isinstance(page_data, JBDetectionResult):
                    result = page_data
                elif isinstance(page_data, (tuple, list)):
                    result = JBDetectionResult.from_tuple(page_data)
                else:
                    continue

                # ── Multiple-JB page skip ───────────────────────
                if len(result.jb_identifiers) > 1:
                    page_warnings.append({
                        "type": "PAGE_SKIPPED_MULTIPLE_JB",
                        "item": ",".join(sorted(str(j) for j in result.jb_identifiers)),
                        "pages": [page_num],
                        "tag_count": len(result.tags),
                        "severity": "ERROR",
                        "description": (
                            f"Page {page_num} of '{pdf_name}' was SKIPPED because "
                            f"multiple JB identifiers were detected: "
                            f"{sorted(str(j) for j in result.jb_identifiers)}. "
                            f"{len(result.tags)} tags on this page were NOT exported."
                        ),
                        "action": (
                            "Manually review the page and choose the correct JB, "
                            "or split the page so it contains a single JB."
                        ),
                        "pdf_name": pdf_name,
                    })
                    continue

                # ── JB-not-found warning ────────────────────────
                _jb_value = list(result.jb_identifiers)[0] if result.jb_identifiers else ""
                _jb_not_found = False
                if not result.jb_identifiers and (result.tags or result.spare_identifiers):
                    _jb_not_found = True
                    _jb_value = f"JB_NOT_FOUND (page {page_num})"
                    page_warnings.append({
                        "type": "JB_NOT_FOUND",
                        "item": _jb_value,
                        "pages": [page_num],
                        "tag_count": len(result.tags) + len(result.spare_identifiers),
                        "severity": "WARNING",
                        "description": (
                            f"Page {page_num} of '{pdf_name}' contains "
                            f"{len(result.tags)} tag(s) and "
                            f"{len(result.spare_identifiers)} spare(s), but NO JB "
                            f"identifier could be detected."
                        ),
                        "action": "Check the page header (top of page).",
                        "pdf_name": pdf_name,
                    })

                # Pick a single MC for this page
                mc_list = list(result.mc_identifiers)
                jb_list = list(result.jb_identifiers)
                selected_mc = self._select_best_mc(mc_list, jb_list)
                cable_desc = self._select_best_cable_description(result.cable_descriptions)
                raw_cable_desc = self._select_best_cable_description(result.raw_cable_descriptions) or cable_desc

                # ── Tags ────────────────────────────────────────
                for tag in sorted(result.tags):
                    tag_number = (result.tag_to_number.get(tag)
                                   or master_tag_numbers.get(tag))
                    if not tag_number:
                        # Skip tags without a number — same as original.
                        continue
                    terminal_info = self.generate_terminal_numbers(tag_number)
                    wire_str = self.generate_mc_wire_colors_enhanced(tag_number)
                    wire_parts = [p.strip() for p in wire_str.split(",")]
                    wire_code_1 = wire_parts[0] if len(wire_parts) > 0 else ""
                    wire_code_2 = wire_parts[1] if len(wire_parts) > 1 else ""

                    # Match status
                    info = result.tag_match_info.get(tag)
                    if info is None and isinstance(result.tag_match_info, dict):
                        # tag_match_info might be a raw dict (legacy path)
                        raw_info = result.tag_match_info.get(tag, {})
                        if isinstance(raw_info, dict):
                            info = TagMatchInfo(
                                match_type=raw_info.get("match_type", "unmatched"),
                                score=float(raw_info.get("score", 0.0) or 0.0),
                                ocr_text=str(raw_info.get("ocr_text", "")),
                                matched_tag=str(raw_info.get("matched_tag", "")),
                                bbox=tuple(raw_info.get("bbox", (0, 0, 0, 0))),
                                reason=str(raw_info.get("reason", "")),
                            )
                    match_status = "Assigned"
                    if info is not None:
                        if info.match_type == "exact":
                            match_status = f"Exact Match (score: {info.score:.3f})"
                        elif info.match_type == "similar":
                            match_status = f"Similar Match (score: {info.score:.3f})"
                        elif info.match_type == "unmatched":
                            match_status = "Unmatched"

                    row_warning = ""
                    if _jb_not_found:
                        row_warning = (
                            f"JB_NOT_FOUND: page {page_num} has tag {tag} but no "
                            f"JB was detected on this page. Tag was NOT assigned to any JB."
                        )

                    new_df_data.append({
                        "PDF_Name": pdf_name,
                        "Page": page_num,
                        "JB": _jb_value,
                        "MC": selected_mc,
                        "Tag/SPARE": tag,
                        "Tag_Number": tag_number,
                        "Wire_Code_1": wire_code_1,
                        "Wire_Code_2": wire_code_2,
                        "Terminal_First_Number": terminal_info["terminal_first"],
                        "Terminal_Second_Number": terminal_info["terminal_second"],
                        "Cable_Code": cable_desc,
                        "SCR_Terminal_Number": terminal_info["scr_terminal"],
                        "Cable_Description": raw_cable_desc,
                        "Type": "Tag",
                        "Tag_Number_Status": match_status,
                        "Warning": row_warning,
                    })

                # ── SPAREs ──────────────────────────────────────
                spare_prefix = (self.spare_examples or "SPARE").strip().upper()
                for spare_idx, spare in enumerate(result.spare_identifiers):
                    spare_id = f"{spare_prefix}_{spare_idx + 1}"
                    spare_number = (result.tag_to_number.get(spare_id)
                                     or master_tag_numbers.get(spare_id))
                    if not spare_number:
                        # Auto-assign after the last tag number
                        max_existing = max(result.tag_to_number.values(), default=0)
                        spare_number = max_existing + spare_idx + 1
                    terminal_info = self.generate_terminal_numbers(spare_number)
                    wire_str = self.generate_mc_wire_colors_enhanced(spare_number)
                    wire_parts = [p.strip() for p in wire_str.split(",")]
                    wire_code_1 = wire_parts[0] if len(wire_parts) > 0 else ""
                    wire_code_2 = wire_parts[1] if len(wire_parts) > 1 else ""

                    row_warning = ""
                    if _jb_not_found:
                        row_warning = (
                            f"JB_NOT_FOUND: page {page_num} has SPARE {spare} but no "
                            f"JB was detected on this page. SPARE was NOT assigned to any JB."
                        )

                    spare_status = (
                        "Assigned (Position-based)"
                        if (result.tag_to_number.get(spare_id)
                            or master_tag_numbers.get(spare_id))
                        else "Auto-assigned (WARNING: not in tag_to_number)"
                    )
                    new_df_data.append({
                        "PDF_Name": pdf_name,
                        "Page": page_num,
                        "JB": _jb_value,
                        "MC": selected_mc,
                        "Tag/SPARE": spare,
                        "Tag_Number": spare_number,
                        "Wire_Code_1": wire_code_1,
                        "Wire_Code_2": wire_code_2,
                        "Terminal_First_Number": terminal_info["terminal_first"],
                        "Terminal_Second_Number": terminal_info["terminal_second"],
                        "Cable_Code": cable_desc,
                        "SCR_Terminal_Number": terminal_info["scr_terminal"],
                        "Cable_Description": raw_cable_desc,
                        "Type": "SPARE",
                        "Tag_Number_Status": spare_status,
                        "Warning": row_warning,
                    })

        # Build DataFrame with the EXACT column order
        if new_df_data:
            df = pd.DataFrame(new_df_data)
            for col in INTERMEDIATE_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            df = df[INTERMEDIATE_COLUMNS]
            df = df.sort_values(["PDF_Name", "Page", "Tag_Number"],
                                 na_position="last")
        else:
            df = pd.DataFrame(columns=INTERMEDIATE_COLUMNS)

        # Save
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".",
                    exist_ok=True)
        df.to_excel(output_path, index=False)
        logger.info("Intermediate Excel saved: %s (%d rows)",
                     output_path, len(df))

        # Stash warnings for later
        self._last_page_warnings = page_warnings
        return df

    def _select_best_mc(self, mc_identifiers: List[str],
                          jb_identifiers: List[str]) -> str:
        """Pick the best MC identifier for a page.

        Delegates to :class:`PatternMatcher` if available; otherwise
        just returns the first MC.
        """
        if not mc_identifiers:
            return ""
        if hasattr(self, "_pattern_matcher") and self._pattern_matcher is not None:
            return self._pattern_matcher.select_best_mc_identifier(
                mc_identifiers, jb_identifiers,
            )
        return str(mc_identifiers[0])

    def set_pattern_matcher(self, matcher: Any) -> None:
        """Allow the facade to inject a PatternMatcher for best-MC selection."""
        self._pattern_matcher = matcher  # type: ignore[attr-defined]

    # ── Final Excel (IO List enriched) ─────────────────────────────
    def create_final_excel(
        self,
        intermediate_path: str,
        io_list_path: str,
        output_path: str,
        all_ocr_tags: Optional[Set[str]] = None,
    ) -> Tuple["Any", List[str], List[str]]:
        """Merge the intermediate Excel with the IO List.

        Returns ``(final_df, unmatched_io_tags, unmatched_pdf_tags)``.
        """
        import pandas as pd  # type: ignore

        try:
            intermediate_df = pd.read_excel(intermediate_path)
            io_list_df = pd.read_excel(io_list_path)
        except Exception as exc:
            logger.error("Error reading input Excels: %s", exc)
            return pd.DataFrame(), [], []

        io_col = self._config.excel_io_list_tag_column
        if io_col not in io_list_df.columns:
            # Try common alternatives
            for alt in ("Tag", "Tag No", "Tag No.", "tag", "tag no",
                         "TAG NO", "TAG NO.", "TAG"):
                if alt in io_list_df.columns:
                    io_col = alt
                    break

        inter_tag_col = self._config.excel_intermediate_tag_column

        # Build UPPER → original-case maps
        inter_upper_to_orig = {
            str(v).strip().upper(): str(v).strip()
            for v in intermediate_df[inter_tag_col] if pd.notna(v)
        }
        io_upper_to_orig = {
            str(v).strip().upper(): str(v).strip()
            for v in io_list_df[io_col] if pd.notna(v)
        }
        inter_tags_upper = set(inter_upper_to_orig.keys())
        io_tags_upper = set(io_upper_to_orig.keys())

        # OCR tags (may include unmatched candidates)
        ocr_upper: Set[str] = set()
        if all_ocr_tags:
            ocr_upper = {str(t).strip().upper() for t in all_ocr_tags if t}
        else:
            ocr_upper = set(inter_tags_upper)

        # Filter out NC* tokens (cable codes, not tags)
        ocr_upper = {t for t in ocr_upper if not t.startswith("NC")}

        # ── Fuzzy matching OCR tags → IO List tags ─────────────────
        ocr_to_io_map: Dict[str, str] = {}
        matched_io_tags: Set[str] = set()

        try:
            import Levenshtein as _lev  # type: ignore
            _fuzzy = True
        except Exception:
            _fuzzy = False

        # Character confusion pairs (for OCR-error correction)
        char_confusions = [
            ("V", "Y"), ("Y", "V"),
            ("S", "5"), ("5", "S"),
            ("O", "0"), ("0", "O"),
            ("B", "8"), ("8", "B"),
            ("G", "6"), ("6", "G"),
            ("Z", "2"), ("2", "Z"),
            ("I", "1"), ("1", "I"),
            ("D", "0"), ("0", "D"),
        ]

        for ocr_tag in ocr_upper:
            if ocr_tag.startswith("NC"):
                continue
            # 1. Exact
            if ocr_tag in io_tags_upper:
                ocr_to_io_map[ocr_tag] = ocr_tag
                matched_io_tags.add(ocr_tag)
                continue
            if not _fuzzy:
                continue
            # 2. Levenshtein fuzzy
            best_io = ""
            best_score = 0.0
            for io_tag in io_tags_upper:
                score = _lev.ratio(ocr_tag, io_tag)
                if score > best_score:
                    best_score = score
                    best_io = io_tag
            if best_io and best_score >= self._config.match_levenshtein_threshold:
                ocr_to_io_map[ocr_tag] = best_io
                matched_io_tags.add(best_io)
                continue
            # 3. Single-char substitution
            if best_score < self._config.match_levenshtein_threshold:
                for old_ch, new_ch in char_confusions:
                    positions = [j for j, c in enumerate(ocr_tag) if c == old_ch]
                    for pos in positions:
                        cand = ocr_tag[:pos] + new_ch + ocr_tag[pos + 1:]
                        for io_tag in io_tags_upper:
                            score = _lev.ratio(cand, io_tag)
                            if score > best_score:
                                best_score = score
                                best_io = io_tag
                if best_io and best_score >= self._config.match_levenshtein_threshold:
                    ocr_to_io_map[ocr_tag] = best_io
                    matched_io_tags.add(best_io)

        # Unmatched
        unmatched_pdf_tags_upper = set(ocr_upper) - set(ocr_to_io_map.keys())
        unmatched_pdf_tags_upper = {t for t in unmatched_pdf_tags_upper
                                     if not t.startswith("NC")}
        unmatched_io_tags_upper = io_tags_upper - matched_io_tags

        # ── Build final DataFrame ─────────────────────────────────
        final_df = io_list_df.copy()
        intermediate_cols_to_add = [c for c in intermediate_df.columns
                                     if c != inter_tag_col]
        for col in intermediate_cols_to_add:
            if col not in final_df.columns:
                final_df[col] = None

        # Helper column for joining
        intermediate_df = intermediate_df.copy()
        intermediate_df["_TAG_UPPER_HELPER_"] = intermediate_df[inter_tag_col].apply(
            lambda x: str(x).strip().upper() if pd.notna(x) else ""
        )

        pdf_to_io_map = inter_tags_upper & io_tags_upper
        for ocr_tag_upper, io_tag_upper in ocr_to_io_map.items():
            if ocr_tag_upper != io_tag_upper:
                if ocr_tag_upper in inter_tags_upper:
                    pdf_to_io_map.add(ocr_tag_upper)
                if io_tag_upper not in pdf_to_io_map:
                    pdf_to_io_map.add(io_tag_upper)

        for idx, row in final_df.iterrows():
            io_tag = (str(row[io_col]).strip().upper()
                       if pd.notna(row[io_col]) else "")
            if not io_tag:
                continue
            # 1. Exact match in intermediate
            if io_tag in pdf_to_io_map:
                match_row = intermediate_df[
                    intermediate_df["_TAG_UPPER_HELPER_"] == io_tag
                ]
                if not match_row.empty:
                    src = match_row.iloc[0]
                    for col in intermediate_cols_to_add:
                        final_df.at[idx, col] = src.get(col, None)
                    continue
            # 2. Fuzzy-matched OCR tags
            matched_ocr_tags = [o for o, i in ocr_to_io_map.items() if i == io_tag]
            for ocr_tag in matched_ocr_tags:
                if ocr_tag in inter_tags_upper:
                    match_row = intermediate_df[
                        intermediate_df["_TAG_UPPER_HELPER_"] == ocr_tag
                    ]
                    if not match_row.empty:
                        src = match_row.iloc[0]
                        for col in intermediate_cols_to_add:
                            final_df.at[idx, col] = src.get(col, None)
                        break
                else:
                    for col in intermediate_cols_to_add:
                        if col == "Type":
                            final_df.at[idx, col] = "Tag"
                        elif col == "Tag_Number_Status":
                            final_df.at[idx, col] = "Matched (Fuzzy OCR)"
                    break
            # 3. Fuzzy matched but no intermediate data
            if io_tag in matched_io_tags:
                for col in intermediate_cols_to_add:
                    if col == "Type" and pd.isna(final_df.at[idx, col]):
                        final_df.at[idx, col] = "Tag"
                    elif col == "Tag_Number_Status" and pd.isna(final_df.at[idx, col]):
                        final_df.at[idx, col] = "Matched (Fuzzy OCR)"

        # JB_SPARE_COUNT column (used by some downstream UIs)
        if "JB" in intermediate_df.columns:
            type_upper = (intermediate_df.get("Type", pd.Series([""] * len(intermediate_df)))
                           .astype(str).str.strip().str.upper())
            tag_upper = (intermediate_df.get(inter_tag_col,
                                              pd.Series([""] * len(intermediate_df)))
                          .astype(str).str.strip().str.upper())
            spare_mask = type_upper.eq("SPARE") | tag_upper.str.contains("SPARE", na=False)
            jb_norm = intermediate_df["JB"].astype(str).str.strip().str.upper()
            spare_counts = (
                intermediate_df.loc[spare_mask]
                .assign(_JB_NORM=jb_norm[spare_mask])
                .groupby("_JB_NORM")
                .size()
                .to_dict()
            )
            jb_col_final = "JB" if "JB" in final_df.columns else None
            if jb_col_final:
                final_df["JB_SPARE_COUNT"] = final_df[jb_col_final].apply(
                    lambda jb: int(spare_counts.get(str(jb).strip().upper(), 0))
                )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".",
                    exist_ok=True)
        final_df.to_excel(output_path, index=False)
        logger.info("Final Excel saved: %s (%d rows)", output_path, len(final_df))

        # Convert unmatched back to original case
        unmatched_pdf_tags_original: List[str] = []
        for tag_upper in unmatched_pdf_tags_upper:
            if all_ocr_tags:
                for ocr_tag in all_ocr_tags:
                    if str(ocr_tag).strip().upper() == tag_upper:
                        unmatched_pdf_tags_original.append(str(ocr_tag).strip())
                        break
                else:
                    unmatched_pdf_tags_original.append(tag_upper)
            else:
                unmatched_pdf_tags_original.append(tag_upper)

        unmatched_io_tags_original = [
            io_upper_to_orig.get(t, t) for t in unmatched_io_tags_upper
        ]

        return (final_df,
                sorted(unmatched_io_tags_original),
                sorted(unmatched_pdf_tags_original))

    # ── Unmatched Excel ────────────────────────────────────────────
    def create_unmatched_excel(
        self,
        unmatched_pdf_tags: List[str],
        unmatched_io_tags: List[str],
        output_path: str,
    ) -> None:
        """Create the unmatched-tags report."""
        import pandas as pd  # type: ignore

        data: List[Dict[str, Any]] = []
        for tag in sorted(unmatched_pdf_tags):
            data.append({
                "Tag": tag,
                "Source": "PDF (OCR)",
                "Status": "Found in PDF but not in IO List",
                "Severity": "WARNING",
                "Action": "Verify tag correctness or add to IO List",
                "Match_Type": "Not Matched",
            })
        for tag in sorted(unmatched_io_tags):
            data.append({
                "Tag": tag,
                "Source": "IO List",
                "Status": "In IO List but not found in PDF",
                "Severity": "INFO",
                "Action": "Check if tag should appear in PDF",
                "Match_Type": "N/A",
            })
        if data:
            df = pd.DataFrame(data, columns=UNMATCHED_COLUMNS)
        else:
            df = pd.DataFrame([{
                "Tag": "N/A",
                "Source": "N/A",
                "Status": "All tags matched successfully!",
                "Severity": "SUCCESS",
                "Action": "No action needed",
                "Match_Type": "N/A",
            }], columns=UNMATCHED_COLUMNS)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".",
                    exist_ok=True)
        df.to_excel(output_path, index=False)
        logger.info("Unmatched Excel saved: %s (%d rows)",
                     output_path, len(df))

    # ── Warnings Excel ─────────────────────────────────────────────
    def create_warnings_excel(self,
                                warnings: List[Dict[str, Any]],
                                output_path: str) -> None:
        """Create the warnings report Excel."""
        import pandas as pd  # type: ignore

        if warnings:
            rows = []
            for w in warnings:
                pages = w.get("pages", [])
                if pages and all(isinstance(p, int) for p in pages):
                    pages_str = ", ".join(str(p) for p in sorted(pages))
                else:
                    pages_str = ", ".join(str(p) for p in pages)
                rows.append({
                    "Warning_Type": w.get("type", ""),
                    "Item": w.get("item", ""),
                    "PDF_Name": w.get("pdf_name", ""),
                    "Pages": pages_str,
                    "Tag_Count": w.get("tag_count", 0),
                    "Severity": w.get("severity", ""),
                    "Description": w.get("description", ""),
                    "Action": w.get("action", ""),
                })
            df = pd.DataFrame(rows, columns=WARNINGS_COLUMNS)
            severity_order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
            df["_sev_order"] = df["Severity"].map(severity_order).fillna(3)
            df = df.sort_values(
                by=["_sev_order", "Warning_Type", "Item"]
            ).drop(columns=["_sev_order"])
        else:
            df = pd.DataFrame([{
                "Warning_Type": "NONE",
                "Item": "N/A",
                "PDF_Name": "N/A",
                "Pages": "N/A",
                "Tag_Count": 0,
                "Severity": "SUCCESS",
                "Description": "No JB-not-found, duplicate-JB, or duplicate-tag issues detected.",
                "Action": "No action needed.",
            }], columns=WARNINGS_COLUMNS)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".",
                    exist_ok=True)
        df.to_excel(output_path, index=False)
        logger.info("Warnings Excel saved: %s (%d rows)",
                     output_path, len(df))


__all__ = [
    "ExcelExporter",
    "INTERMEDIATE_COLUMNS",
    "UNMATCHED_COLUMNS",
    "WARNINGS_COLUMNS",
]
