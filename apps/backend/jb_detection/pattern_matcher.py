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
    WIRE_COLOR_PATTERN,
)
from .models import JBDetectionResult, OcrDetection, TagMatchInfo
from .extraction_logger import log_extraction
from .legacy_candidates import LegacyCandidateRules
from .tag_matcher import _lev

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
        self._explicit_patterns = {
            "jb": jb_examples is not None, "mc": mc_examples is not None,
            "spare": spare_examples is not None, "cable": cable_examples is not None,
        }

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
        self.io_tag_matcher: Optional[Any] = None

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
            self._explicit_patterns["jb"] = True
            self.jb_examples_list = _parse_multi_patterns(jb_examples)
            self.jb_examples = (
                ",".join(self.jb_examples_list) if self.jb_examples_list else None
            )
        if mc_examples is not None:
            self._explicit_patterns["mc"] = True
            self.mc_examples_list = _parse_multi_patterns(mc_examples)
            self.mc_examples = (
                ",".join(self.mc_examples_list) if self.mc_examples_list else None
            )
        if spare_examples is not None:
            self._explicit_patterns["spare"] = True
            self.spare_examples_list = _parse_multi_patterns(spare_examples)
            self.spare_examples = (
                ",".join(self.spare_examples_list) if self.spare_examples_list else None
            )
        if cable_examples is not None:
            self._explicit_patterns["cable"] = True
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
        separator = r"[\s._-]*"
        def prefix_pattern(prefix):
            return separator.join(re.escape(part) for part in re.findall(r"[A-Z]+|\d+", prefix))
        def identifier_pattern(prefixes):
            alternatives = []
            for prefix in prefixes:
                pieces = re.findall(r"[A-Z]+|\d+", prefix)
                if prefix.isalpha():
                    alternatives.append(rf"(?:\d{{1,4}}{separator})?{prefix_pattern(prefix)}")
                elif len(pieces) == 2 and pieces[0].isdigit() and pieces[1].isalpha():
                    alternatives.append(rf"(?:{re.escape(pieces[0])}{separator})?{re.escape(pieces[1])}")
                else:
                    alternatives.append(prefix_pattern(prefix))
            alt = "|".join(alternatives)
            # Alpha segments after the configured prefix must be separated by
            # identifier punctuation. This avoids treating prose such as
            # "JB To ITR ... 100" as one long identifier. A serial may follow
            # the prefix directly or be separated by whitespace/punctuation.
            suffix = (
                rf"(?:[._-]*\d{{1,6}}|\s+\d{{1,6}}"
                rf"|(?:[._-]+[A-Z0-9]{{1,8}})+[._-]*\d{{1,6}})"
                rf"(?:[._-]+[A-Z0-9]{{1,8}})*"
            )
            return re.compile(rf"(?<![A-Z0-9])((?:{alt}){suffix})(?![A-Z0-9])", re.IGNORECASE)
        if self.jb_examples_list:
            self.jb_regex = identifier_pattern(self.jb_examples_list)
        else:
            self.jb_regex = re.compile(r"(?!)") if self._explicit_patterns["jb"] else JB_PATTERN
        if self.mc_examples_list:
            self.mc_regex = identifier_pattern(self.mc_examples_list)
        else:
            self.mc_regex = re.compile(r"(?!)") if self._explicit_patterns["mc"] else MC_PATTERN

        # SPARE regex — literal word (optionally with index)
        if self.spare_examples_list:
            alt = "|".join(re.escape(p) + ("S?" if p.upper() == "SPARE" else "")
                           for p in self.spare_examples_list)
            self.spare_regex = re.compile(rf"\b({alt})\b", re.IGNORECASE)
        else:
            self.spare_regex = re.compile(r"(?!)") if self._explicit_patterns["spare"] else SPARE_PATTERN
        if self._explicit_patterns["cable"] and not self.cable_examples:
            self.cable_regex = re.compile(r"(?!)")
        elif self.cable_examples:
            # Treat configured examples as format hints. In particular, a
            # unit-only example (e.g. 12P) describes a cable shape, not a
            # literal brand prefix that should be baked into the detector.
            units = {"P": r"P(?:AIR)?", "PAIR": r"P(?:AIR)?",
                     "T": r"T(?:RIPLE)?", "TRIPLE": r"T(?:RIPLE)?",
                     "C": r"C(?:ORE)?", "CORE": r"C(?:ORE)?"}
            unit_patterns = []
            for example in re.split(r"[,;]", self.cable_examples):
                unit_match = re.search(r"\d+\s*(PAIR|TRIPLE|CORE|P|T|C)\b", example, re.I)
                if unit_match:
                    unit_patterns.append(units[unit_match.group(1).upper()])
            if unit_patterns:
                unit_alt = "|".join(sorted(set(unit_patterns), key=len, reverse=True))
                self.cable_regex = re.compile(
                    rf"(?<![A-Z0-9])([A-Z]{{2,8}}[-_ ]*)?\d{{1,3}}\s*(?:{unit_alt})"
                    rf"(?:\s*(?:X|×)\s*\d+(?:\.\d+)?\s*(?:MM\s*\^?\s*2?)?)?"
                    rf"(?![A-Z0-9])", re.I)
            else:
                self.cable_regex = CABLE_PATTERN
        else:
            self.cable_regex = CABLE_PATTERN

        logger.debug(
            "Regex patterns compiled: JB=%s, MC=%s, SPARE=%s",
            bool(self.jb_examples_list), bool(self.mc_examples_list),
            bool(self.spare_examples_list),
        )

    # ── Token classification ───────────────────────────────────────
    def _is_jb_token(self, text: str) -> bool:
        return bool(self.jb_regex.fullmatch(str(text).strip()))

    def _is_mc_token(self, text: str) -> bool:
        return bool(self.mc_regex.fullmatch(str(text).strip()))

    def _is_spare_token(self, text: str) -> bool:
        t = str(text).upper().strip()
        if not t:
            return False
        return bool(self.spare_regex.search(t))

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

        # Cable codes — ONLY the tight cable pattern (with unit prefixes).
        # The old pattern was so loose it matched "FIT-100-14" as a cable!
        if self._is_cable_token(t):
            return True

        # Wire color codes (BK01, WT12, RD03, …) — use WIRE_COLOR_PATTERN.
        if WIRE_COLOR_PATTERN.match(t):
            return True

        # Preserve the legacy exclusions when structural discovery is broad.
        if re.fullmatch(r"\d{1,3}(?:BK|WH|RD|BL|GN|YL|OR|GY|VI|BN|PK)", t):
            return True
        if re.fullmatch(r"SCR[-_]?\d*", t):
            return True
        for rule in (self.wire_color_rule, self.scr_number_rule):
            for part in re.split(r"[,;\s]+", rule or ""):
                prefix = re.match(r"^([A-Za-z]{2,})", part)
                if prefix and t.startswith(prefix.group(1).upper()) and any(char.isdigit() for char in t):
                    return True

        # Pure numbers (terminal numbers, page numbers)
        if re.fullmatch(r"\d{1,4}", t):
            return True

        # Single letters / very short tokens
        if len(re.sub(r"[\s_-]+", "", t)) < 4 and not (
                self.io_tag_matcher and self.io_tag_matcher.exact_reference(t)):
            return True

        return False

    def _extract_tag(self, text: str) -> str:
        if self.io_tag_matcher is not None:
            candidates = self.io_tag_matcher.extract_candidates(text)
            return next((tag for tag in candidates if not self._is_non_tag_pattern(tag)), "")
        match = self.tag_regex.search(text)
        return match.group(1).upper() if match else ""

    def _looks_like_tag(self, token: str) -> bool:
        """Heuristic: does this token look like an instrument tag?"""
        if not token or self._is_non_tag_pattern(token):
            return False
        t = str(token).strip().upper()
        # Tag regex requires letter(s) + digit(s) minimum.
        if not self._extract_tag(t):
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

        Tags are numbered 1, 2, 3, ... from top of page.
        SPAREs are numbered separately continuing after the last tag number.
        """
        all_items: List[Dict[str, Any]] = []
        
        # Tags first
        for item in tags_with_positions:
            tag_name = str(item.get("tag", ""))
            # Skip cables — they shouldn't be numbered as tags
            if self._is_cable_token(tag_name):
                continue
            all_items.append({
                "name": tag_name,
                "y_position": int(item.get("y", 0)),
                "x_position": int(item.get("x", 0)),
                "type": "tag",
            })
        
        # Sort tags by position (top→bottom, left→right)
        all_items.sort(key=lambda x: (x["y_position"], x["x_position"]))
        
        # Assign numbers 1, 2, 3, ...
        tag_to_number: Dict[str, int] = {}
        canonical_numbers: Dict[str, int] = {}
        tag_num = 1
        for item in all_items:
            if item["type"] == "tag":
                # Preserve OCR spelling in result keys, but assign separator
                # and whitespace variants one shared position number.
                canonical = re.sub(r"\s+", "", item["name"].upper()).replace("_", "-").strip("-.")
                if canonical not in canonical_numbers:
                    canonical_numbers[canonical] = tag_num
                    tag_num += 1
                tag_to_number[item["name"]] = canonical_numbers[canonical]
        
        # SPAREs — numbered separately, continuing after tags
        if spare_identifiers_with_positions:
            spare_items = []
            for idx, item in enumerate(spare_identifiers_with_positions):
                spare_text = str(item.get("id", f"SPARE_{idx + 1}"))
                spare_items.append({
                    "name": spare_text,
                    "y_position": int(item.get("y", 0)),
                    "x_position": int(item.get("x", 0)),
                    "type": "spare",
                })
            spare_items.sort(key=lambda x: (x["y_position"], x["x_position"]))
            
            spare_num = tag_num  # Continue after last tag number
            for item in spare_items:
                if item["name"] not in tag_to_number:
                    tag_to_number[item["name"]] = spare_num
                    spare_num += 1
        
        return tag_to_number

    # ── Best-MC / Best-Cable selection ─────────────────────────────
    @staticmethod
    def select_best_cable_description(cable_descriptions: List[str]) -> str:
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
    def _join_tag_fragments(self, detections: List[OcrDetection]) -> List[OcrDetection]:
        """Join neighboring prefix/serial fragments in OCR and native PDFs."""
        matcher = self.io_tag_matcher
        if matcher is None:
            return detections
        joined = []
        index = 0
        while index < len(detections):
            first = detections[index]
            group = [first]
            combined = first.text
            consumed = 1
            if not matcher.extract_candidates(first.text):
                for following in detections[index + 1:index + 8]:
                    previous = group[-1]
                    overlap = min(previous.y + previous.height, following.y + following.height) - max(previous.y, following.y)
                    gap = following.x - (previous.x + previous.width)
                    if (overlap < min(previous.height, following.height) * 0.5
                            or gap < -2 or gap > max(previous.height, following.height) * 1.5):
                        break
                    if matcher.extract_candidates(following.text):
                        break
                    group.append(following)
                    combined += " " + following.text
                    if matcher.matches_io_pattern(combined):
                        consumed = len(group)
                        x = min(item.x for item in group)
                        y = min(item.y for item in group)
                        right = max(item.x + item.width for item in group)
                        bottom = max(item.y + item.height for item in group)
                        first = OcrDetection(combined, min(item.confidence for item in group), [], (x, y, right - x, bottom - y))
                        break
            joined.append(first)
            index += consumed
        return joined

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
        cable_candidates: List[Tuple[str, str, OcrDetection]] = []

        # Track positions for numbering
        tags_with_positions: List[Dict[str, Any]] = []
        spare_with_positions: List[Dict[str, Any]] = []

        # Track which bboxes we've already used for a given tag
        seen_tag_bbox: Dict[str, BBox_T] = {}  # type: ignore

        source_detections = self._join_tag_fragments(detections)
        io_references = getattr(self.io_tag_matcher, "reference_tags", []) if self.io_tag_matcher else []
        profile_candidates = (
            LegacyCandidateRules(io_references, _lev.distance)
            if io_references else None
        )
        for det in source_detections:
            text = (det.text or "").strip()
            if not text:
                continue
            text_upper = text.upper()

            # ── JB ───────────────────────────────────────────────
            if self.jb_regex.search(text):
                # Use the regex match to extract the canonical JB id
                m = self.jb_regex.search(text)
                jb_id = m.group(0).upper() if m else text_upper
                jb_id = _normalize_code_token(jb_id)
                for configured_prefix in self.jb_examples_list:
                    parts = re.findall(r"[A-Z]+|\d+", configured_prefix)
                    if (len(parts) == 2 and parts[0].isdigit() and parts[1].isalpha()
                            and jb_id.startswith(parts[1]) and not jb_id.startswith(parts[0])):
                        jb_id = parts[0] + jb_id
                        break
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
                    log_extraction(
                        "classification",
                        page=0, source="",
                        text=text, bbox=det.bbox, confidence=det.confidence,
                        category="JB", reason="matched JB_PATTERN",
                        pattern_name="JB_PATTERN", pattern_match=jb_id,
                    )
                continue

            # ── MC ───────────────────────────────────────────────
            if self.mc_regex.search(text):
                m = self.mc_regex.search(text)
                mc_id = m.group(0).upper() if m else text_upper
                mc_id = _normalize_code_token(mc_id)
                if mc_id:
                    mc_identifiers.add(mc_id)
                    # Store MC in tag_match_info so the annotator can draw it
                    tag_match_info[mc_id] = TagMatchInfo(
                        match_type="MC",
                        score=det.confidence,
                        ocr_text=text,
                        matched_tag=mc_id,
                        bbox=det.bbox,
                        reason="MC identifier",
                    )
                    log_extraction(
                        "classification",
                        page=0, source="",
                        text=text, bbox=det.bbox, confidence=det.confidence,
                        category="MC", reason="matched MC_PATTERN",
                        pattern_name="MC_PATTERN", pattern_match=mc_id,
                    )
                continue

            # ── SPARE ────────────────────────────────────────────
            if self._is_spare_token(text):
                spare_match = self.spare_regex.search(text)
                spare_id = spare_match.group(0).upper() if spare_match else "SPARE"
                if spare_id == "SPARES":
                    spare_id = "SPARE"
                before = text[:spare_match.start()] if spare_match else ""
                count_match = re.search(r"(?:^|\s)(\d{1,2})\s*$", before)
                count = int(count_match.group(1)) if count_match else 1
                if not count_match:
                    nearby = []
                    for other in source_detections:
                        if other is det or not re.fullmatch(r"\d{1,2}", other.text.strip()):
                            continue
                        overlap = min(det.y + det.height, other.y + other.height) - max(det.y, other.y)
                        gap = det.x - (other.x + other.width)
                        if overlap >= min(det.height, other.height) * 0.5 and 0 <= gap <= det.height * 1.5:
                            nearby.append((gap, int(other.text.strip())))
                    if nearby:
                        count = min(nearby)[1]
                for _ in range(count):
                    spare_identifiers.append(spare_id)
                    occurrence_id = f"SPARE_{len(spare_identifiers)}"
                    spare_with_positions.append({
                        "id": occurrence_id, "bbox": det.bbox, "spare": spare_id,
                        "y": det.bbox[1], "x": det.bbox[0],
                    })
                    tag_match_info[occurrence_id] = TagMatchInfo(
                        match_type="SPARE", score=det.confidence, ocr_text=text,
                        matched_tag=spare_id, bbox=det.bbox,
                        reason="SPARE count label" if count > 1 else "SPARE identifier",
                    )
                log_extraction(
                    "classification",
                    page=0, source="",
                    text=text, bbox=det.bbox, confidence=det.confidence,
                    category="SPARE", reason="matched SPARE_PATTERN",
                    pattern_name="SPARE_PATTERN", pattern_match=spare_id,
                )
                continue

            # ── Tag (instrument tag) — checked BEFORE cable! ─────
            # CRITICAL: the old order checked cable BEFORE tag, which
            # caused "FIT-100-14" to be misclassified as a cable.
            # The new order checks tag first, so instrument tags are
            # always captured before the cable fallback.
            io_candidates = (
                self.io_tag_matcher.extract_candidates(text)
                if self.io_tag_matcher is not None else
                ([self._extract_tag(text)] if self._looks_like_tag(text) else [])
            )
            recovered_candidates = []
            if profile_candidates is not None:
                io_candidate_keys = {
                    self.io_tag_matcher.separator_key(candidate)
                    for candidate in io_candidates
                }
                recovered_candidates = [
                    (candidate, score)
                    for candidate, score in profile_candidates.extract_candidates(text)
                    if (
                        self.io_tag_matcher.separator_key(candidate) not in io_candidate_keys
                        and not self._is_non_tag_pattern(candidate)
                    )
                ]

            if io_candidates or recovered_candidates:
                # The profile fallback only retains a review candidate; it
                # never assigns it to an IO reference or changes its OCR text.
                recovered_by_tag = {candidate: score for candidate, score in recovered_candidates}
                candidate_tags = list(dict.fromkeys(io_candidates + list(recovered_by_tag)))
                for tag in candidate_tags:
                    if self._is_non_tag_pattern(tag):
                        continue
                    tag = tag.strip().upper() if self.io_tag_matcher is not None else _normalize_code_token(tag)
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
                        match_type=("unmatched_candidate" if tag in recovered_by_tag else "unmatched"),
                        score=0.0,
                        ocr_text=text,
                        matched_tag="",
                        bbox=det.bbox,
                        reason=(
                            f"IO-independent OCR profile candidate (shape score {recovered_by_tag[tag]:.2f}); "
                            "not matched to an IO tag; review required."
                            if tag in recovered_by_tag else "Awaiting IO List match"
                        ),
                    )
                    log_extraction(
                        "classification",
                        page=0, source="",
                        text=text, bbox=det.bbox, confidence=det.confidence,
                        category="Tag", reason="matched TAG_PATTERN",
                        pattern_name="TAG_PATTERN", pattern_match=tag,
                    )
                continue

            # ── Cable description (checked AFTER tag) ─────────────
            # Only tokens that did NOT look like tags reach here.
            cable_match = self.cable_regex.search(text)
            if cable_match:
                cable_desc = cable_match.group(0).upper().strip()
                if cable_desc:
                    cable_descriptions.append(cable_desc)
                    raw_cable_descriptions.append(text_upper)
                    cable_candidates.append((cable_desc, text_upper, det))
                    # Store the cable's position so the annotator can
                    # draw a bounding box for it.
                    tags_with_positions.append({
                        "tag": cable_desc,
                        "y": det.bbox[1],
                        "x": det.bbox[0],
                    })
                    tag_match_info[cable_desc] = TagMatchInfo(
                        match_type="Cable",
                        score=det.confidence,
                        ocr_text=text,
                        matched_tag=cable_desc,
                        bbox=det.bbox,
                        reason="Cable description",
                    )
                    log_extraction(
                        "classification",
                        page=0, source="",
                        text=text, bbox=det.bbox, confidence=det.confidence,
                        category="Cable", reason="matched CABLE_PATTERN",
                        pattern_name="CABLE_PATTERN", pattern_match=cable_desc,
                    )
                continue

            # ── Unknown (did not match any category) ───────────────
            # Log it so the LLM agent can later learn new patterns.
            if len(text) >= 3:
                log_extraction(
                    "classification",
                    page=0, source="",
                    text=text, bbox=det.bbox, confidence=det.confidence,
                    category="Unknown",
                    reason="did not match any pattern (JB/MC/Tag/Cable/SPARE)",
                )

        # When a cable format was supplied, associate the description with
        # the nearest MC on the page. This mirrors the former local search and
        # prevents a remote cable label from winning solely by digit size.
        if self.cable_examples and mc_identifiers and cable_candidates:
            mc_boxes = [info.bbox for key, info in tag_match_info.items()
                        if info.match_type == "MC" and info.bbox]
            selected: Dict[int, Tuple[str, str, OcrDetection]] = {}
            for mc_box in mc_boxes:
                mx, my, mw, mh = mc_box
                nearby = []
                for item in cable_candidates:
                    cable_box = item[2].bbox
                    cx, cy, cw, ch = cable_box
                    dx = abs((cx + cw / 2) - (mx + mw / 2))
                    dy = abs((cy + ch / 2) - (my + mh / 2))
                    if dx <= max(180, mw * 8) and dy <= max(120, mh * 8):
                        nearby.append((dx + 3 * dy, item))
                if nearby:
                    _, best = min(nearby, key=lambda pair: pair[0])
                    selected[id(best[2])] = best
            if selected:
                cable_descriptions = [item[0] for item in selected.values()]
                raw_cable_descriptions = [item[1] for item in selected.values()]
                selected_ids = {item[0] for item in selected.values()}
                for cable_desc, _, _ in cable_candidates:
                    if cable_desc not in selected_ids:
                        tag_match_info.pop(cable_desc, None)

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
