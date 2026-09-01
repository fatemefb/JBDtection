"""JBDetection — pattern matcher.

Given a list of :class:`OcrDetection` from PaddleOCR, classify each
detection as one of:

- **JB** identifier (junction box)
- **MC** identifier (motor cable)
- **Tag** (instrument tag — TIT-101, FCV-101-A, UZSO-2482, …)
- **Cable** description (NC-0-1-2-C-3-BL style multi-segment codes)
- **SPARE** keyword

The matcher is **stateful** — it holds the regex patterns compiled from
user-supplied examples (``jb_examples``, ``mc_examples``, etc.) and
recompiles them when :meth:`set_patterns` is called.

Tag numbering
-------------
Tags and SPAREs are numbered by vertical position (top-to-bottom,
left-to-right within a row) — same as the original code. The result is
returned as :class:`JBDetectionResult` which preserves the legacy
9-tuple structure via :meth:`JBDetectionResult.to_tuple`.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from .config import (
    CABLE_PATTERN, INSTRUMENT_PREFIXES, JB_PATTERN, MC_PATTERN,
    SPARE_PATTERN, STOP_WORDS, TAG_PATTERN,
)
from .models import JBDetectionResult, OcrDetection, TagMatchInfo

logger = logging.getLogger("jb_detection.pattern_matcher")


# ── Helpers ─────────────────────────────────────────────────────────────
def _parse_multi_patterns(value: Any) -> List[str]:
    """Parse a comma/space/newline-separated list of prefixes.

    Accepts strings ("JSF,JSX,JSY"), lists of strings, or nested lists.
    Returns: list of uppercased non-empty strings.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items: List[str] = []
        for v in value:
            items.extend(
                str(v).replace("\n", ",").replace(" ", ",").split(",")
            )
    else:
        items = str(value).replace("\n", ",").replace(" ", ",").split(",")
    return [i.strip().upper() for i in items if i.strip()]


def _normalize_code_token(token: Any) -> str:
    """Normalize an OCR token to an uppercase code-style identifier."""
    if token is None:
        return ""
    s = str(token).strip().upper()
    if not s:
        return ""
    s = re.sub(r"[^A-Z0-9._-]", "", s)
    return s.strip("._-")


def _is_prefixed_identifier(token: str, prefix: str,
                              require_digit: bool = True) -> bool:
    """True if ``token`` looks like ``<prefix><digits/sep>...``.

    Reduces false-positives — e.g. prevents 'ATACHONICAL' from matching
    when ``prefix='IC'``.
    """
    if not token or not prefix:
        return False
    prefix = str(prefix).strip().upper()
    token = str(token).strip().upper()
    if not token.startswith(prefix):
        return False
    if len(token) <= len(prefix):
        return False
    if require_digit and not any(ch.isdigit() for ch in token):
        return False
    # If prefix ends with alnum, require separator or digit next
    if prefix[-1].isalnum() and len(token) > len(prefix):
        next_ch = token[len(prefix)]
        if next_ch.isalpha():
            return False
    return True


# ── PatternMatcher ──────────────────────────────────────────────────────
class PatternMatcher:
    """Classify OCR detections into JB / MC / Tag / Cable / SPARE.

    Constructor accepts the same example lists as the original
    :meth:`TagJBExtractor.set_patterns` for backward compatibility.
    """

    def __init__(self,
                 jb_examples: Optional[Union[str, List[str]]] = None,
                 mc_examples: Optional[Union[str, List[str]]] = None,
                 spare_examples: Optional[Union[str, List[str]]] = None,
                 cable_examples: Optional[Union[str, List[str]]] = None,
                 wire_color_rule: Optional[str] = None,
                 scr_number_rule: Optional[str] = None) -> None:
        # Multi-pattern lists (the canonical form)
        self.jb_examples_list: List[str] = _parse_multi_patterns(jb_examples)
        self.mc_examples_list: List[str] = _parse_multi_patterns(mc_examples)
        self.spare_examples_list: List[str] = _parse_multi_patterns(spare_examples)

        # Backward-compatible comma-joined strings (used by logging/excel)
        self.jb_examples: Optional[str] = (
            ",".join(self.jb_examples_list) if self.jb_examples_list else None
        )
        self.mc_examples: Optional[str] = (
            ",".join(self.mc_examples_list) if self.mc_examples_list else None
        )
        self.spare_examples: Optional[str] = (
            ",".join(self.spare_examples_list) if self.spare_examples_list else None
        )

        if isinstance(cable_examples, list):
            self.cable_examples: Optional[str] = ", ".join(cable_examples)
        elif isinstance(cable_examples, str):
            self.cable_examples = cable_examples.strip() or None
        else:
            self.cable_examples = None

        self.wire_color_rule: Optional[str] = wire_color_rule
        self.scr_number_rule: Optional[str] = scr_number_rule

        # Compiled regexes
        self.jb_regex: Optional[re.Pattern] = None
        self.mc_regex: Optional[re.Pattern] = None
        self.spare_regex: Optional[re.Pattern] = None
        self.cable_regex: re.Pattern = CABLE_PATTERN
        self.tag_regex: re.Pattern = TAG_PATTERN

        self._compile_regex_patterns()

    # ── Configuration ──────────────────────────────────────────────
    def set_patterns(self,
                     jb_examples: Optional[Union[str, List[str]]] = None,
                     mc_examples: Optional[Union[str, List[str]]] = None,
                     spare_examples: Optional[Union[str, List[str]]] = None,
                     cable_examples: Optional[Union[str, List[str]]] = None,
                     wire_color_rule: Optional[str] = None,
                     scr_number_rule: Optional[str] = None) -> None:
        """Update patterns. Only non-None arguments are applied."""
        if jb_examples is not None:
            self.jb_examples_list = _parse_multi_patterns(jb_examples)
            self.jb_examples = (
                ",".join(self.jb_examples_list) if self.jb_examples_list else None
            )
        if mc_examples is not None:
            self.mc_examples_list = _parse_multi_patterns(mc_examples)
            self.mc_examples = (
                ",".join(self.mc_examples_list) if self.mc_examples_list else None
            )
        if spare_examples is not None:
            self.spare_examples_list = _parse_multi_patterns(spare_examples)
            self.spare_examples = (
                ",".join(self.spare_examples_list) if self.spare_examples_list else None
            )
        if cable_examples is not None:
            if isinstance(cable_examples, list):
                self.cable_examples = ", ".join(cable_examples) or None
            else:
                self.cable_examples = str(cable_examples).strip() or None
        if wire_color_rule is not None:
            self.wire_color_rule = wire_color_rule
        if scr_number_rule is not None:
            self.scr_number_rule = scr_number_rule
        self._compile_regex_patterns()

    def set_wire_color_rule(self, rule: Optional[str]) -> None:
        self.wire_color_rule = rule

    def set_scr_number_rule(self, rule: Optional[str]) -> None:
        self.scr_number_rule = rule

    def set_terminal_wire_patterns(self, config: Dict[str, Any]) -> None:
        """Backward-compatible hook for the original API.

        The original code stored ``terminal_pattern`` and
        ``wire_color_rule`` together in a dict. We just forward them
        to the relevant setters.
        """
        if "wire_color_pattern" in config:
            self.wire_color_rule = config["wire_color_pattern"]
        # ``terminal_pattern`` is consumed by the facade's
        # ``generate_terminal_numbers``; we don't store it here.

    def _compile_regex_patterns(self) -> None:
        """Compile regex patterns from the configured example lists."""
        # JB regex — alternation of all prefixes
        if self.jb_examples_list:
            alt = "|".join(re.escape(p) for p in self.jb_examples_list)
            self.jb_regex = re.compile(rf"\b({alt})[-_]?\d+\b", re.IGNORECASE)
        else:
            self.jb_regex = JB_PATTERN

        # MC regex — alternation of all prefixes
        if self.mc_examples_list:
            alt = "|".join(re.escape(p) for p in self.mc_examples_list)
            self.mc_regex = re.compile(rf"\b({alt})[-_]?\d+\b", re.IGNORECASE)
        else:
            self.mc_regex = MC_PATTERN

        # SPARE regex — literal word (optionally with index)
        if self.spare_examples_list:
            alt = "|".join(re.escape(p) for p in self.spare_examples_list)
            self.spare_regex = re.compile(rf"\b({alt})(?:\s*\d+)?\b", re.IGNORECASE)
        else:
            self.spare_regex = SPARE_PATTERN

        logger.debug(
            "Regex patterns compiled: JB=%s, MC=%s, SPARE=%s",
            bool(self.jb_examples_list), bool(self.mc_examples_list),
            bool(self.spare_examples_list),
        )

    # ── Token classification ───────────────────────────────────────
    def _is_jb_token(self, text: str) -> bool:
        t = str(text).upper().strip()
        if not self.jb_examples_list:
            return bool(JB_PATTERN.fullmatch(t))
        return any(_is_prefixed_identifier(t, p, require_digit=False)
                   for p in self.jb_examples_list)

    def _is_mc_token(self, text: str) -> bool:
        t = str(text).upper().strip()
        if not self.mc_examples_list:
            return bool(MC_PATTERN.fullmatch(t))
        return any(_is_prefixed_identifier(t, p, require_digit=False)
                   for p in self.mc_examples_list)

    def _is_spare_token(self, text: str) -> bool:
        t = str(text).upper().strip()
        if not t:
            return False
        if self.spare_examples_list:
            return any(re.search(rf"\b{re.escape(p)}\b", t, re.IGNORECASE)
                       for p in self.spare_examples_list)
        return bool(SPARE_PATTERN.search(t))

    def _is_cable_token(self, text: str) -> bool:
        t = str(text).upper().strip()
        if not t:
            return False
        return bool(self.cable_regex.search(t))

    def _is_non_tag_pattern(self, token: str) -> bool:
        """True = token should NOT be treated as a tag.

        Catches: JB/MC/SPARE identifiers, cable codes, wire color codes,
        pure stop words.
        """
        if not token:
            return True
        t = str(token).strip().upper()

        # Stop words (Page, Sheet, BK, WT, …)
        if t in STOP_WORDS:
            return True

        # JB / MC / SPARE
        if self._is_jb_token(t):
            return True
        if self._is_mc_token(t):
            return True
        if self._is_spare_token(t):
            return True

        # Cable codes (NC-0-1-2-C-3-BL)
        if self._is_cable_token(t):
            return True

        # Wire color codes (BK01, WT12, RD03, …)
        if re.fullmatch(r"(BK|WT|RD|BL|GN|YL|BR|GR|OG|PK|PR)\d{1,4}", t):
            return True

        # Pure numbers (terminal numbers, page numbers)
        if re.fullmatch(r"\d{1,4}", t):
            return True

        # Single letters / very short tokens
        if len(t) < 3:
            return True

        return False

    def _looks_like_tag(self, token: str) -> bool:
        """Heuristic: does this token look like an instrument tag?"""
        if not token or self._is_non_tag_pattern(token):
            return False
        t = str(token).strip().upper()
        # Tag regex requires letter(s) + digit(s) minimum.
        if not self.tag_regex.search(t):
            return False
        # Reject if it doesn't contain at least one digit (tags always do)
        if not any(c.isdigit() for c in t):
            return False
        return True

    # ── Tag-number assignment ──────────────────────────────────────
    def assign_tag_numbers_by_position(
        self,
        tags_with_positions: List[Dict[str, Any]],
        spare_identifiers_with_positions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, int]:
        """Number tags and SPAREs by vertical position (top→bottom).

        Same algorithm as the original ``assign_tag_numbers_by_position``:
        combine tags + spares, sort by ``(y, x)``, assign 1-based numbers.
        SPARE ids are generated as ``f"{spare_examples}_{idx+1}"``.
        """
        all_items: List[Dict[str, Any]] = []
        for item in tags_with_positions:
            all_items.append({
                "name": str(item.get("tag", "")),
                "y_position": int(item.get("y", 0)),
                "x_position": int(item.get("x", 0)),
                "type": "tag",
            })
        if spare_identifiers_with_positions:
            spare_prefix = (self.spare_examples or "SPARE").strip().upper()
            for idx, item in enumerate(spare_identifiers_with_positions):
                spare_id = f"{spare_prefix}_{idx + 1}"
                all_items.append({
                    "name": spare_id,
                    "y_position": int(item.get("y", 0)),
                    "x_position": int(item.get("x", 0)),
                    "type": "spare",
                    "original_text": str(item.get("spare", "SPARE")),
                })
        if not all_items:
            return {}
        all_items.sort(key=lambda x: (x["y_position"], x["x_position"]))
        return {item["name"]: idx for idx, item in enumerate(all_items, start=1)}

    # ── Best-MC / Best-Cable selection ─────────────────────────────
    def select_best_cable_description(self, cable_descriptions: List[str]) -> str:
        """Pick the cable code with the largest sum of digits.

        Real cable codes (NC-12-3-4-A-5-WHT) typically use larger
        numbers than header/legend samples (NC-0-1-2-C-3-BL).
        """
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

    def select_best_mc_identifier(self,
                                    mc_identifiers: Iterable[str],
                                    jb_identifiers: Iterable[str]) -> str:
        """Pick a single best MC for a page in a deterministic way.

        Prefers codes that start with one of the configured MC prefixes
        and look structurally valid. When a JB exists, prefers an MC
        whose suffix matches the JB suffix (e.g. JB-EEV-101 → IC-EEV-101).
        """
        mc_prefixes = list(self.mc_examples_list)
        if not mc_prefixes:
            return ""

        raw_candidates = list(mc_identifiers) if mc_identifiers else []
        norm_all = [_normalize_code_token(c) for c in raw_candidates]
        norm_all = [c for c in norm_all if c]

        # Pass 1: strict (prefix + digit)
        norm = [c for c in norm_all
                if any(_is_prefixed_identifier(c, p, require_digit=True)
                       for p in mc_prefixes)]
        # Pass 2: relax digit requirement
        if not norm:
            norm = [c for c in norm_all
                    if any(_is_prefixed_identifier(c, p, require_digit=False)
                           for p in mc_prefixes)]
        if not norm:
            return ""

        # JB prefix list (multi-pattern)
        jb_prefixes = list(self.jb_examples_list)

        # Build expected_mc using the first MC prefix + JB suffix
        expected_mc: Optional[str] = None
        jb_list = list(jb_identifiers) if jb_identifiers else []
        if jb_list and jb_prefixes:
            jb_norm = _normalize_code_token(jb_list[0])
            if jb_norm:
                for jp in jb_prefixes:
                    if jb_norm.startswith(jp) and len(jb_norm) > len(jp):
                        jb_suffix = jb_norm[len(jp):]
                        expected_mc = mc_prefixes[0] + jb_suffix
                        break

        def candidate_score(cand: str) -> Tuple[float, int, int, int]:
            digits = sum(ch.isdigit() for ch in cand)
            seps = cand.count("-") + cand.count("_") + cand.count(".")
            length_penalty = -len(cand)
            similarity = 0.0
            if expected_mc:
                try:
                    import Levenshtein  # type: ignore
                    similarity = float(Levenshtein.ratio(cand, expected_mc))
                except Exception:
                    similarity = 0.0
            return (similarity, digits, seps, length_penalty)

        norm_sorted = sorted(set(norm))
        return max(norm_sorted, key=lambda c: (candidate_score(c), c))

    # ── Main entry point ───────────────────────────────────────────
    def match(self, detections: List[OcrDetection]) -> JBDetectionResult:
        """Classify OCR detections into the 9-tuple structure.

        Steps
        -----
        1. Walk detections in reading order (already sorted by detector).
        2. For each detection:
           - Check JB / MC / SPARE / Cable regex.
           - If none match and it looks like a tag, collect it.
        3. Build ``all_ocr_tags`` — every plausible tag-like token,
           including unmatched candidates (preserves original behavior).
        4. Number tags + spares by vertical position.
        5. Return :class:`JBDetectionResult`.
        """
        tags: Set[str] = set()
        jb_identifiers: Set[str] = set()
        mc_identifiers: Set[str] = set()
        cable_descriptions: List[str] = []
        spare_identifiers: List[str] = []
        raw_cable_descriptions: List[str] = []
        all_ocr_tags: Set[str] = set()
        tag_match_info: Dict[str, TagMatchInfo] = {}

        # Track positions for numbering
        tags_with_positions: List[Dict[str, Any]] = []
        spare_with_positions: List[Dict[str, Any]] = []

        # Track which bboxes we've already used for a given tag
        seen_tag_bbox: Dict[str, BBox_T] = {}  # type: ignore

        for det in detections:
            text = (det.text or "").strip()
            if not text:
                continue
            text_upper = text.upper()

            # ── JB ───────────────────────────────────────────────
            if self._is_jb_token(text):
                # Use the regex match to extract the canonical JB id
                m = self.jb_regex.search(text)
                jb_id = m.group(0).upper() if m else text_upper
                jb_id = _normalize_code_token(jb_id)
                if jb_id:
                    jb_identifiers.add(jb_id)
                    tag_match_info[jb_id] = TagMatchInfo(
                        match_type="JB",
                        score=det.confidence,
                        ocr_text=text,
                        matched_tag=jb_id,
                        bbox=det.bbox,
                        reason="JB identifier",
                    )
                continue

            # ── MC ───────────────────────────────────────────────
            if self._is_mc_token(text):
                m = self.mc_regex.search(text)
                mc_id = m.group(0).upper() if m else text_upper
                mc_id = _normalize_code_token(mc_id)
                if mc_id:
                    mc_identifiers.add(mc_id)
                # MC tokens might also contain a cable description —
                # fall through to cable check below.

            # ── Cable description ────────────────────────────────
            cable_match = self.cable_regex.search(text)
            if cable_match:
                cable_desc = cable_match.group(1).upper()
                if cable_desc:
                    cable_descriptions.append(cable_desc)
                    raw_cable_descriptions.append(text_upper)
                # A cable code is not a tag — skip the tag branch.
                continue

            # ── SPARE ────────────────────────────────────────────
            if self._is_spare_token(text):
                spare_match = self.spare_regex.search(text)
                spare_id = spare_match.group(0).upper() if spare_match else "SPARE"
                spare_identifiers.append(spare_id)
                spare_with_positions.append({
                    "spare": spare_id,
                    "y": det.bbox[1],
                    "x": det.bbox[0],
                })
                tag_match_info[spare_id] = TagMatchInfo(
                    match_type="SPARE",
                    score=det.confidence,
                    ocr_text=text,
                    matched_tag=spare_id,
                    bbox=det.bbox,
                    reason="SPARE identifier",
                )
                continue

            # ── Tag (instrument tag) ─────────────────────────────
            if self._looks_like_tag(text):
                # Try to extract the canonical tag from the text
                tag_match = self.tag_regex.search(text)
                tag = tag_match.group(1).upper() if tag_match else text_upper
                tag = _normalize_code_token(tag)
                if not tag:
                    continue

                tags.add(tag)
                all_ocr_tags.add(tag)
                tags_with_positions.append({
                    "tag": tag,
                    "y": det.bbox[1],
                    "x": det.bbox[0],
                })
                seen_tag_bbox[tag] = det.bbox
                # Initial tag_match_info — match_type will be updated by
                # TagMatcher later.
                tag_match_info[tag] = TagMatchInfo(
                    match_type="unmatched",
                    score=0.0,
                    ocr_text=text,
                    matched_tag="",
                    bbox=det.bbox,
                    reason="Awaiting IO List match",
                )

        # ── Number tags + spares by position ─────────────────────
        tag_to_number = self.assign_tag_numbers_by_position(
            tags_with_positions, spare_with_positions,
        )

        return JBDetectionResult(
            tags=tags,
            jb_identifiers=jb_identifiers,
            mc_identifiers=mc_identifiers,
            cable_descriptions=cable_descriptions,
            spare_identifiers=spare_identifiers,
            tag_to_number=tag_to_number,
            raw_cable_descriptions=raw_cable_descriptions,
            tag_match_info=tag_match_info,
            all_ocr_tags=all_ocr_tags,
            tag_positions={item["tag"]: (item["y"], item["x"])
                            for item in tags_with_positions},
            spare_positions=spare_with_positions,
        )


# Type alias used internally — kept here so we don't pollute models.py
BBox_T = Tuple[int, int, int, int]


__all__ = ["PatternMatcher"]
