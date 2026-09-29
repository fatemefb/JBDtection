"""JBDetection — Pattern configuration (pluggable, LLM-friendly).

This module defines the **pattern catalogue** used by
:class:`PatternMatcher` to classify OCR detections into
JB / MC / Tag / Cable / SPARE / Unknown.

Design goals
------------
1. **All patterns in one place** — easy to audit, easy to override.
2. **Pluggable** — patterns can be injected at runtime via
   :func:`inject_patterns` (e.g. by an LLM agent that learns them
   from a sample PDF).
3. **Documented decisions** — every regex has a comment explaining
   what it matches and what it intentionally does NOT match.
4. **No false positives on instrument tags** — the legacy
   ``CABLE_PATTERN`` was so loose it matched ``FIT-100-14`` as a
   cable. The new patterns are tighter and tag-matching takes
   priority over cable-matching.

ISA-5.1 instrument tag format
-----------------------------
A standard instrument tag has the form:

    <prefix>-<loop>-<suffix>

where:
  • prefix  = 2-5 letters (e.g. FIT, PDIT, TIT, LDIT, LIT)
  • loop    = 3-6 digits, optionally split by a hyphen (e.g. 100-14)
  • suffix  = optional letter (e.g. A, B, C)

Examples of VALID tags:
    FIT-100-14       (Flow Indication Transmitter, loop 100-14)
    PDIT-100-11      (Pressure Differential Indication Transmitter)
    LDIT-100-04      (Level Differential Indication Transmitter)
    LIT-100-02       (Level Indication Transmitter)
    TIT-100-03       (Temperature Indication Transmitter)
    TE-5223          (Temperature Element)
    PT-1014-A        (Pressure Transmitter, suffix A)

Examples of NON-tags (must NOT match TAG_PATTERN):
    BK01, WT02       (wire color codes — too short, prefix not in INSTRUMENT_PREFIXES)
    FRT-01Px1.5mm²   (cable spec — has "mm²" and unusual segments)
    JB-DIA-100-001   (JB identifier — handled by JB_PATTERN)
    MC-DIA-100-001   (MC identifier — handled by MC_PATTERN)
    01, 02, SCR      (terminal numbers / labels)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern

logger = logging.getLogger("jb_detection.pattern_config")


# ═══════════════════════════════════════════════════════════════════════
# ISA-5.1 instrument tag prefixes (comprehensive)
# ═══════════════════════════════════════════════════════════════════════
# Source: ISA-5.1 standard + common project-specific extensions.
# These are the prefixes that a tag is allowed to start with.
# A token that looks like a tag (letter+digits) but whose prefix is
# NOT in this list will be classified as "Unknown" rather than "Tag".
INSTRUMENT_PREFIXES: List[str] = [
    # ── Pressure ───────────────────────────────────────────────────────
    "PT", "PI", "PDT", "PDI", "PIT", "PS", "PDS", "PC", "PDC", "PR", "PRT",
    "PSV", "PRV", "PSE", "PDSE",
    # ── Temperature ────────────────────────────────────────────────────
    "TE", "TT", "TI", "TIT", "TS", "TDS", "TC", "TDC", "TR", "TRT", "TSE",
    "TDIT", "TDI", "TIC",
    # ── Flow ───────────────────────────────────────────────────────────
    "FE", "FT", "FI", "FIT", "FS", "FDS", "FC", "FDC", "FR", "FRT", "FSE",
    "FCV", "FOV", "FV", "FIC", "FQI", "FQIT",
    # ── Level ──────────────────────────────────────────────────────────
    "LE", "LT", "LI", "LIT", "LS", "LDS", "LC", "LDC", "LR", "LRT", "LSE",
    "LDIT", "LDI", "LIC",
    # ── Analysis ──────────────────────────────────────────────────────
    "AE", "AT", "AI", "AIT", "AS", "ADS", "AC", "ADC", "AR", "ART", "ASE",
    # ── Control / Actuation (valves, etc.) ────────────────────────────
    "FCV", "FOV", "FV", "LCV", "LOV", "LV", "PCV", "POV", "PV",
    "TCV", "TOV", "TV", "HV", "SV", "XV", "FY", "LY", "PY", "TY", "HY",
    # ── Safety / Interlock ────────────────────────────────────────────
    "ZSO", "ZSC", "ZS", "ZI", "ZIT", "PSV", "PRV", "TSV", "FSV", "BSD", "BD",
    # ── Logic / Sequence (SIS) ─────────────────────────────────────────
    "UZSO", "UZSC", "UZS", "KI", "KIT", "KC", "KDC",
    # ── Miscellaneous ─────────────────────────────────────────────────
    "HS", "HI", "HIT", "HC", "HDC", "SP", "XT", "XI", "XIT", "YC",
    "YIC", "YIT", "YS", "YSD", "YSL",
    "FRT",  # Fire/Flame Retardant Transmitter (project-specific)
]


# ═══════════════════════════════════════════════════════════════════════
# Pattern definitions
# ═══════════════════════════════════════════════════════════════════════

# ── TAG_PATTERN ─────────────────────────────────────────────────────────
# Matches ISA-5.1 instrument tags: <prefix>-<loop>[-<suffix>]
# Examples: FIT-100-14, PDIT-100-11, TE-5223, PT-1014-A
#
# CRITICAL: this must NOT match wire codes (BK01, WT02), cable specs
# (FRT-01Px1.5mm²), or JB/MC identifiers.
#
# The pattern requires:
#   • 2-5 letter prefix (upper or lower)
#   • hyphen separator
#   • 3-6 digit loop number (optionally split: 100-14)
#   • optional letter suffix (A, B, C, etc.)
#
# Capture group 1 = the full tag (used by PatternMatcher.match()).
TAG_PATTERN: Pattern = re.compile(
    r"\b([A-Z]{2,5}-\d{2,4}(?:-\d{1,4})?(?:-[A-Z])?)\b",
    re.IGNORECASE,
)

# ── JB_PATTERN (default — user can override via set_patterns) ───────────
# Matches JB identifiers like:
#   JB-101, JB-DIA-100-001, JB100, JB_101
# The default pattern is permissive; user-supplied examples take
# priority (see PatternMatcher._compile_regex_patterns).
JB_PATTERN: Pattern = re.compile(
    r"\b(JB[-_]?[A-Z]*[-_]?\d{1,6}(?:[-_]?\d{1,4})?)\b",
    re.IGNORECASE,
)

# ── MC_PATTERN (default) ────────────────────────────────────────────────
# Matches MC (multi-cable) identifiers like:
#   MC-200, MC-DIA-100-001, MC200, MC_200
MC_PATTERN: Pattern = re.compile(
    r"\b(MC[-_]?[A-Z]*[-_]?\d{1,6}(?:[-_]?\d{1,4})?)\b",
    re.IGNORECASE,
)

# ── SPARE_PATTERN ───────────────────────────────────────────────────────
# Matches the word "SPARE" or "SP" (optionally with a number).
# Case-insensitive. Matches both "SPARE" and "Spare" and "spare 1".
SPARE_PATTERN: Pattern = re.compile(
    r"\b(SPARE|SP|Spare)(?:\s*\d+)?\b",
    re.IGNORECASE,
)

# ── CABLE_PATTERN (TIGHTENED — was the root cause of tag misclassification) ──
# Matches cable specifications like:
#   FRT-01Px1.5mm², FRT-12Px0.75mm², NC-0-1-2-C-3-BL
#
# CRITICAL CHANGE: the old pattern was:
#     ([A-Z]{1,3}\d{0,3}(?:[-_][A-Z0-9]{1,4}){2,})
# which matched ANY string with 3+ hyphen-separated alphanumeric
# segments — including "FIT-100-14" (a tag!). The new pattern
# requires the cable spec to contain a unit (mm², Px, pair, core, etc.)
# OR to have a very different structure from a tag.
#
# Specifically:
#   • Must contain "mm" or "Px" or "pair" or "core" or "PR" or "CR"
#     (cable specs always have a unit), OR
#   • Must start with a known cable prefix (NC, FRT, CBL, CAB, WIR)
CABLE_PATTERN: Pattern = re.compile(
    r"\b("
    r"(?:FRT|NC|CBL|CAB|WIR)[-_][A-Z0-9]{1,6}(?:[-_][A-Z0-9]{1,6})*"
    r"(?:\.\d+)?(?:mm²|mm2|Px|PR|CR|pair|core)?"
    r")\b",
    re.IGNORECASE,
)

# ── WIRE_COLOR_PATTERN ──────────────────────────────────────────────────
# Matches wire color codes: BK01, WT02, RD03, BL04, GN05, YL06, etc.
# These are NOT tags and must be filtered out before tag matching.
WIRE_COLOR_PATTERN: Pattern = re.compile(
    r"^(BK|WT|RD|BL|GN|YL|BR|GR|OG|PK|PR|WH|GY)\d{1,4}$",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════
# Stop words — tokens that should NEVER be classified as tags
# ═══════════════════════════════════════════════════════════════════════
STOP_WORDS: set = {
    # Drawing metadata
    "PAGE", "SHEET", "REV", "REVISION", "DATE", "DRAWN", "CHECKED",
    "APPROVED", "SCALE", "SIZE", "DWG", "DWG NO", "DRAWING",
    "PROJECT", "CLIENT", "CONTRACTOR", "ORIGINATOR",
    "AREA", "TRAIN", "UNIT", "DISC", "SEQ", "STATUS", "CLASS",
    "SOURCE", "PHASE", "DOC", "TYPE", "SHT", "SERIAL",
    "PREP", "CHECK", "APP", "APPROVAL", "TITLE",
    # Wire colors (also matched by WIRE_COLOR_PATTERN, but listed here
    # as a safety net for bare color abbreviations without numbers)
    "BK", "WT", "RD", "BL", "GN", "YL", "OR", "BR", "GR", "VI",
    "BLACK", "WHITE", "RED", "BLUE", "GREEN", "YELLOW", "ORANGE",
    "BROWN", "GRAY", "GREY", "VIOLET", "PINK", "PURPLE",
    # System labels
    "DCS", "PLC", "ESD", "F&G", "SIS", "JB", "MC", "SP", "SPARE",
    "TERMINAL", "TERMINALS", "BLOCK", "BOARD", "CARD", "CHANNEL",
    "QTY", "NO", "TYPE", "REF", "NOTE", "NOTES",
    # Common drawing labels
    "TYP", "TYPICAL", "CONT", "CONTINUED", "SECTION", "DETAIL",
    "OSCR", "SCR",  # terminal block labels
    "N/A", "NA", "TBD", "PE",
}


# ═══════════════════════════════════════════════════════════════════════
# OCR confusion pairs (for tag matching fixup)
# ═══════════════════════════════════════════════════════════════════════
# Maps an OCR-misread character to the correct one.
# Used by TagMatcher._try_confusion_fixup() when an exact match fails.
OCR_CONFUSION_PAIRS: Dict[str, str] = {
    "O": "0",
    "o": "0",
    "I": "1",
    "l": "1",
    "S": "5",
    "s": "5",
    "B": "8",
    "Z": "2",
    "z": "2",
    "G": "6",
    "g": "9",
    "D": "0",
    "Q": "0",
    "U": "0",
}


# ═══════════════════════════════════════════════════════════════════════
# Pluggable pattern injection (LLM-friendly)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PatternSet:
    """A complete set of patterns that can be injected at runtime.

    This dataclass is the **contract** between the pattern matcher
    and any external pattern source (LLM agent, config file, etc.).

    An LLM agent that has learned patterns from a sample PDF can
    construct a :class:`PatternSet` and pass it to
    :func:`inject_patterns` to override the defaults.
    """

    tag_pattern: Optional[str] = None       # regex string (not compiled)
    jb_pattern: Optional[str] = None
    mc_pattern: Optional[str] = None
    spare_pattern: Optional[str] = None
    cable_pattern: Optional[str] = None
    wire_color_pattern: Optional[str] = None
    instrument_prefixes: Optional[List[str]] = None
    stop_words: Optional[List[str]] = None
    ocr_confusion_pairs: Optional[Dict[str, str]] = None
    # Metadata about the source (for logging / audit)
    source: str = "default"                # "default" | "llm" | "config_file" | ...
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag_pattern": self.tag_pattern,
            "jb_pattern": self.jb_pattern,
            "mc_pattern": self.mc_pattern,
            "spare_pattern": self.spare_pattern,
            "cable_pattern": self.cable_pattern,
            "wire_color_pattern": self.wire_color_pattern,
            "instrument_prefixes": self.instrument_prefixes,
            "stop_words": self.stop_words,
            "ocr_confusion_pairs": self.ocr_confusion_pairs,
            "source": self.source,
            "notes": self.notes,
        }


# The currently active pattern set (defaults are loaded at import time).
_active_pattern_set: PatternSet = PatternSet(source="default")


def get_active_patterns() -> PatternSet:
    """Return the currently active :class:`PatternSet`."""
    return _active_pattern_set


def inject_patterns(patterns: PatternSet) -> None:
    """Inject a new set of patterns, replacing the defaults.

    This is the **LLM injection point** — an LLM agent that has
    learned patterns from sample PDFs can call this function to
    update what the PatternMatcher uses for classification.

    Parameters
    ----------
    patterns:
        A :class:`PatternSet` with the new patterns. ``None`` fields
        fall back to the defaults.
    """
    global _active_pattern_set, TAG_PATTERN, JB_PATTERN, MC_PATTERN
    global SPARE_PATTERN, CABLE_PATTERN, WIRE_COLOR_PATTERN
    global INSTRUMENT_PREFIXES, STOP_WORDS, OCR_CONFUSION_PAIRS

    _active_pattern_set = patterns

    if patterns.tag_pattern:
        TAG_PATTERN = re.compile(patterns.tag_pattern, re.IGNORECASE)
    if patterns.jb_pattern:
        JB_PATTERN = re.compile(patterns.jb_pattern, re.IGNORECASE)
    if patterns.mc_pattern:
        MC_PATTERN = re.compile(patterns.mc_pattern, re.IGNORECASE)
    if patterns.spare_pattern:
        SPARE_PATTERN = re.compile(patterns.spare_pattern, re.IGNORECASE)
    if patterns.cable_pattern:
        CABLE_PATTERN = re.compile(patterns.cable_pattern, re.IGNORECASE)
    if patterns.wire_color_pattern:
        WIRE_COLOR_PATTERN = re.compile(patterns.wire_color_pattern, re.IGNORECASE)
    if patterns.instrument_prefixes:
        INSTRUMENT_PREFIXES = list(patterns.instrument_prefixes)
    if patterns.stop_words:
        STOP_WORDS = set(patterns.stop_words)
    if patterns.ocr_confusion_pairs:
        OCR_CONFUSION_PAIRS = dict(patterns.ocr_confusion_pairs)

    logger.info(
        "Pattern set injected: source=%s notes=%s | tag=%s jb=%s mc=%s "
        "spare=%s cable=%s prefixes=%d stop_words=%d",
        patterns.source, patterns.notes,
        TAG_PATTERN.pattern, JB_PATTERN.pattern, MC_PATTERN.pattern,
        SPARE_PATTERN.pattern, CABLE_PATTERN.pattern,
        len(INSTRUMENT_PREFIXES), len(STOP_WORDS),
    )


def reset_patterns_to_default() -> None:
    """Reset all patterns to the built-in defaults."""
    global _active_pattern_set
    _active_pattern_set = PatternSet(source="default")
    # Re-import to get the original compiled patterns back.
    # We re-compile them here to be safe.
    global TAG_PATTERN, JB_PATTERN, MC_PATTERN, SPARE_PATTERN
    global CABLE_PATTERN, WIRE_COLOR_PATTERN
    TAG_PATTERN = re.compile(
        r"\b([A-Z]{2,5}-\d{2,4}(?:-\d{1,4})?(?:-[A-Z])?)\b", re.IGNORECASE,
    )
    JB_PATTERN = re.compile(
        r"\b(JB[-_]?[A-Z]*[-_]?\d{1,6}(?:[-_]?\d{1,4})?)\b", re.IGNORECASE,
    )
    MC_PATTERN = re.compile(
        r"\b(MC[-_]?[A-Z]*[-_]?\d{1,6}(?:[-_]?\d{1,4})?)\b", re.IGNORECASE,
    )
    SPARE_PATTERN = re.compile(
        r"\b(SPARE|SP|Spare)(?:\s*\d+)?\b", re.IGNORECASE,
    )
    CABLE_PATTERN = re.compile(
        r"\b((?:FRT|NC|CBL|CAB|WIR)[-_][A-Z0-9]{1,6}(?:[-_][A-Z0-9]{1,6})*"
        r"(?:\.\d+)?(?:mm²|mm2|Px|PR|CR|pair|core)?)\b",
        re.IGNORECASE,
    )
    WIRE_COLOR_PATTERN = re.compile(
        r"^(BK|WT|RD|BL|GN|YL|BR|GR|OG|PK|PR|WH|GY)\d{1,4}$",
        re.IGNORECASE,
    )
    logger.info("Patterns reset to defaults.")


__all__ = [
    "INSTRUMENT_PREFIXES",
    "TAG_PATTERN",
    "JB_PATTERN",
    "MC_PATTERN",
    "SPARE_PATTERN",
    "CABLE_PATTERN",
    "WIRE_COLOR_PATTERN",
    "STOP_WORDS",
    "OCR_CONFUSION_PAIRS",
    "PatternSet",
    "get_active_patterns",
    "inject_patterns",
    "reset_patterns_to_default",
]
