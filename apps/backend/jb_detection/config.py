"""JBDetection — configuration.

This module defines:
  - :class:`Config` — a dataclass holding all runtime settings.
  - :data:`DEFAULT_CONFIG` — the default :class:`Config` instance.
  - :func:`load_config` — reads environment variables (prefix ``JBDET_``)
    and returns a :class:`Config`.
  - Pattern constants used by :mod:`pattern_matcher` and :mod:`tag_matcher`:
    ``JB_PATTERN``, ``MC_PATTERN``, ``TAG_PATTERN``, ``CABLE_PATTERN``,
    ``SPARE_PATTERN``, ``INSTRUMENT_PREFIXES``, ``OCR_CONFUSION_PAIRS``,
    ``STOP_WORDS``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ═══════════════════════════════════════════════════════════════════════════
# Config dataclass
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    """Runtime configuration for the JBDetection pipeline.

    All fields have defaults; the :data:`DEFAULT_CONFIG` singleton is used
    unless :func:`load_config` finds overriding environment variables.
    """

    # ── PaddleOCR ──────────────────────────────────────────────────
    # Production policy: GPU is mandatory. The default is True; at
    # startup validate_gpu_environment() enforces this and aborts if
    # the GPU is unavailable. Set to False ONLY for local development
    # (and even then, prefer JBDET_ALLOW_CPU=1 to keep the policy
    # explicit).
    paddle_use_angle_cls: bool = True
    paddle_lang: str = "en"
    paddle_show_log: bool = False
    paddle_use_gpu: bool = True
    paddle_min_confidence: float = 0.30

    # ── PDF rendering ─────────────────────────────────────────────
    pdf_dpi: int = 300
    pdf_max_pages: int = 500
    pdf_max_file_size_mb: int = 500
    pdf_auto_dpi_enabled: bool = True
    pdf_auto_dpi_threshold_1: int = 100
    pdf_auto_dpi_threshold_2: int = 300
    pdf_auto_dpi_low: int = 200
    pdf_auto_dpi_lower: int = 150
    gc_interval_pages: int = 10

    # ── Digital PDF detection ──────────────────────────────────────
    # A page is considered "digital" if it has >= this many characters
    # of extractable text.
    digital_pdf_min_text_chars: int = 30
    # A page is considered "digital" if text covers >= this fraction
    # of the expected text area (0.0 = disabled).
    digital_pdf_text_coverage_threshold: float = 0.0

    # ── Preprocessing ─────────────────────────────────────────────
    preprocess_clahe_clip: float = 2.0
    preprocess_clahe_tile: int = 8
    preprocess_gaussian_kernel: int = 3
    preprocess_use_otsu: bool = True

    # ── Tag matching ───────────────────────────────────────────────
    match_similar_threshold: float = 0.85
    match_levenshtein_threshold: float = 0.92
    match_use_confusion_pairs: bool = True

    # ── Excel ──────────────────────────────────────────────────────
    # IO List tag column (user-provided Excel)
    excel_io_list_tag_column: str = "Tag No"
    # Intermediate Excel tag column — this MUST match the actual column
    # name in INTERMEDIATE_COLUMNS (see excel_exporter.py:40).
    # The column is "Tag/SPARE" (not "Tag No").
    excel_intermediate_tag_column: str = "Tag/SPARE"

    # ── Visualization ─────────────────────────────────────────────
    # Bounding-box colors (BGR tuples) for each category.
    viz_color_tag: tuple = (0, 200, 0)       # green
    viz_color_jb: tuple = (255, 100, 0)      # blue (BGR)
    viz_color_mc: tuple = (0, 165, 255)      # orange (BGR)
    viz_color_cable: tuple = (0, 255, 255)   # yellow (BGR)
    viz_color_spare: tuple = (128, 128, 128) # gray
    viz_color_unknown: tuple = (0, 0, 255)   # red (BGR)
    viz_box_thickness: int = 2
    viz_label_font_scale: float = 0.5


DEFAULT_CONFIG = Config()


def load_config() -> Config:
    """Load configuration from environment variables.

    Environment variables use the prefix ``JBDET_``. Boolean values are
    ``true``/``false`` (case-insensitive). Numeric values are parsed
    as ``int`` or ``float`` based on the field type.

    Returns
    -------
    Config
        A :class:`Config` instance with values overridden by environment
        variables where present.
    """
    cfg = Config()

    # Boolean fields
    for field_name in ("paddle_use_angle_cls", "paddle_show_log",
                        "paddle_use_gpu", "pdf_auto_dpi_enabled",
                        "preprocess_use_otsu", "match_use_confusion_pairs"):
        env_key = f"JBDET_{field_name.upper()}"
        val = os.environ.get(env_key)
        if val is not None:
            setattr(cfg, field_name, val.strip().lower() in ("1", "true", "yes", "on"))

    # String fields
    for field_name in ("paddle_lang", "excel_io_list_tag_column",
                        "excel_intermediate_tag_column"):
        env_key = f"JBDET_{field_name.upper()}"
        val = os.environ.get(env_key)
        if val is not None:
            setattr(cfg, field_name, val)

    # Integer fields
    for field_name in ("pdf_dpi", "pdf_max_pages", "pdf_max_file_size_mb",
                        "pdf_auto_dpi_threshold_1", "pdf_auto_dpi_threshold_2",
                        "pdf_auto_dpi_low", "pdf_auto_dpi_lower",
                        "gc_interval_pages",
                        "digital_pdf_min_text_chars",
                        "preprocess_clahe_tile", "preprocess_gaussian_kernel"):
        env_key = f"JBDET_{field_name.upper()}"
        val = os.environ.get(env_key)
        if val is not None:
            try:
                setattr(cfg, field_name, int(val))
            except ValueError:
                pass

    # Float fields
    for field_name in ("paddle_min_confidence",
                        "digital_pdf_text_coverage_threshold",
                        "preprocess_clahe_clip",
                        "match_similar_threshold", "match_levenshtein_threshold"):
        env_key = f"JBDET_{field_name.upper()}"
        val = os.environ.get(env_key)
        if val is not None:
            try:
                setattr(cfg, field_name, float(val))
            except ValueError:
                pass

    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# Pattern constants
# ═══════════════════════════════════════════════════════════════════════════

# Instrument tag prefixes (ISA-5.1 style).
# These are used for one-hot encoding in tag vectors and for the default
# tag regex.
INSTRUMENT_PREFIXES: List[str] = [
    # Pressure
    "PT", "PI", "PDT", "PDI", "PIT", "PS", "PDS", "PC", "PDC", "PR", "PRT",
    # Temperature
    "TE", "TT", "TI", "TIT", "TS", "TDS", "TC", "TDC", "TR", "TRT",
    # Flow
    "FE", "FT", "FI", "FIT", "FS", "FDS", "FC", "FDC", "FR", "FRT",
    # Level
    "LE", "LT", "LI", "LIT", "LS", "LDS", "LC", "LDC", "LR", "LRT",
    # Analysis
    "AE", "AT", "AI", "AIT", "AS", "ADS", "AC", "ADC", "AR", "ART",
    # Control / Actuation
    "FCV", "FOV", "FV", "LCV", "LOV", "LV", "PCV", "POV", "PV",
    "TCV", "TOV", "TV", "HV", "SV", "XV",
    # Safety / Interlock
    "ZSO", "ZSC", "ZS", "ZI", "ZIT", "PSV", "PRV", "TSV", "FSV",
    # Logic / Sequence
    "UZSO", "UZSC", "UZS", "KI", "KIT", "KC", "KDC",
    # Miscellaneous
    "HS", "HI", "HIT", "HC", "HDC", "SP", "XT", "XI", "XIT", "YC",
    "YIC", "YIT", "YS", "YSD", "YSL",
]

# Tag pattern: matches ISA-5.1 instrument tags like TE-5223, PT-1014-A, FCV-101.
# Two-three letter prefix, hyphen, digits, optional suffix.
# NOTE: group(1) captures the full tag (required by PatternMatcher.match()).
TAG_PATTERN: re.Pattern = re.compile(
    r"\b([A-Z]{2,5}-\d{2,4}(?:-\d{1,4})?(?:-[A-Z])?)\b",
    re.IGNORECASE,
)

# JB pattern: "JB" prefix + optional letters + digits (e.g. JB-DIA-100-001)
JB_PATTERN: re.Pattern = re.compile(
    r"\b(JB[-_]?[A-Z]*[-_]?\d{1,6}(?:[-_]?\d{1,4})?)\b",
    re.IGNORECASE,
)

# MC pattern: "MC" prefix + optional letters + digits (e.g. MC-DIA-100-001)
MC_PATTERN: re.Pattern = re.compile(
    r"\b(MC[-_]?[A-Z]*[-_]?\d{1,6}(?:[-_]?\d{1,4})?)\b",
    re.IGNORECASE,
)

# SPARE pattern: the word "SPARE" or "SP" (optionally with a number).
SPARE_PATTERN: re.Pattern = re.compile(
    r"\b(SPARE|SP|Spare)(?:\s*\d+)?\b",
    re.IGNORECASE,
)

# Cable pattern: TIGHTENED — was the root cause of tag misclassification.
# Only matches cable specifications with known prefixes (FRT, NC, CBL, etc.)
# or with explicit units (mm², Px, pair, core).
# NOTE: group(1) captures the full cable code (required by PatternMatcher.match()).
CABLE_PATTERN: re.Pattern = re.compile(
    r"\b((?:FRT|NC|CBL|CAB|WIR)[-_][A-Z0-9]{1,6}(?:[-_][A-Z0-9]{1,6})*"
    r"(?:\.\d+)?(?:mm²|mm2|Px|PR|CR|pair|core)?)\b",
    re.IGNORECASE,
)

# Wire color pattern: BK01, WT02, RD03, etc. (NOT tags!)
WIRE_COLOR_PATTERN: re.Pattern = re.compile(
    r"^(BK|WT|RD|BL|GN|YL|BR|GR|OG|PK|PR|WH|GY)\d{1,4}$",
    re.IGNORECASE,
)

# OCR confusion pairs — maps an OCR-misread character to the correct one.
# Used by :meth:`TagMatcher._try_confusion_fixup`.
# Key = what OCR sees, Value = what it probably should be.
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

# Stop words — tokens that should never be classified as tags.
# These are common labels found on engineering drawings that are not
# instrument identifiers.
STOP_WORDS: Set[str] = {
    "PAGE", "SHEET", "REV", "REVISION", "DATE", "DRAWN", "CHECKED",
    "APPROVED", "SCALE", "SIZE", "DWG", "DWG NO", "DRAWING",
    "BK", "WT", "RD", "BL", "GN", "YL", "OR", "BR", "GR", "VI",
    "BLACK", "WHITE", "RED", "BLUE", "GREEN", "YELLOW", "ORANGE",
    "BROWN", "GRAY", "GREY", "VIOLET",
    "DCS", "PLC", "ESD", "F&G", "SIS",
    "NO", "TYPE", "QTY", "REF", "NOTE", "NOTES",
    "TERMINAL", "TERMINALS", "BLOCK", "BOARD",
    "CHANNEL", "CH", "CARD",
    "PROJECT", "CLIENT", "CONTRACTOR",
    "N/A", "NA", "TBD",
    "TYP", "TYPICAL",
    "CONT", "CONTINUED",
    "SECTION", "DETAIL",
}


__all__ = [
    "Config",
    "DEFAULT_CONFIG",
    "load_config",
    "INSTRUMENT_PREFIXES",
    "TAG_PATTERN",
    "JB_PATTERN",
    "MC_PATTERN",
    "SPARE_PATTERN",
    "CABLE_PATTERN",
    "WIRE_COLOR_PATTERN",
    "OCR_CONFUSION_PAIRS",
    "STOP_WORDS",
]
