"""JBDetection — unified PDF + OCR + pattern-matching pipeline.

Public API
----------
- :class:`TagJBExtractor` — the workhorse facade (wraps everything).
- :class:`DataAnalysis` — thin wrapper with optional PDF classifier.
- :class:`UnifiedPdfProcessor` — unified PDF processor (digital + scanned).
- :class:`JBDetectionPipeline` — low-level pipeline (single-page).
- :class:`TextDetector` — PaddleOCR singleton wrapper.
- :class:`PatternMatcher` — classify detections into JB/MC/Tag/Cable/SPARE.
- :class:`TagMatcher` — fuzzy-match tags against IO List.
- :class:`ExcelExporter` — generate Excel outputs.
- :class:`PDFAnnotator` — generate color-coded annotated PDFs.

Models
------
- :class:`OcrDetection` — single text region (text, confidence, polygon, bbox).
- :class:`TagMatchInfo` — match result for one tag.
- :class:`JBDetectionResult` — per-page result (9-tuple backward compat).
- :class:`PageResult` — page result + metadata.
"""

from __future__ import annotations

# Config
from .config import (
    Config,
    DEFAULT_CONFIG,
    load_config,
    INSTRUMENT_PREFIXES,
    TAG_PATTERN,
    JB_PATTERN,
    MC_PATTERN,
    SPARE_PATTERN,
    CABLE_PATTERN,
    OCR_CONFUSION_PAIRS,
    STOP_WORDS,
)

# Models
from .models import (
    BBox,
    Polygon,
    OcrDetection,
    TagMatchInfo,
    JBDetectionResult,
    PageResult,
)

# Pipeline components
from .detector import TextDetector, DetectorError
from .pattern_matcher import PatternMatcher
from .tag_matcher import TagMatcher
from .pipeline import JBDetectionPipeline

# Unified PDF processor (new — handles both digital + scanned PDFs)
from .pdf_type_detector import PdfTypeDetector, PdfType
from .digital_text_extractor import DigitalTextExtractor
from .unified_pdf_processor import UnifiedPdfProcessor

# Visualization
from .annotator import PDFAnnotator

# Facade (optional — requires excel_exporter which requires pandas/openpyxl)
try:
    from .excel_exporter import ExcelExporter
    from .facade import TagJBExtractor, DataAnalysis
    _HAS_EXCEL_EXPORTER = True
except ImportError:
    _HAS_EXCEL_EXPORTER = False

__all__ = [
    # Config
    "Config",
    "DEFAULT_CONFIG",
    "load_config",
    "INSTRUMENT_PREFIXES",
    "TAG_PATTERN",
    "JB_PATTERN",
    "MC_PATTERN",
    "SPARE_PATTERN",
    "CABLE_PATTERN",
    "OCR_CONFUSION_PAIRS",
    "STOP_WORDS",
    # Models
    "BBox",
    "Polygon",
    "OcrDetection",
    "TagMatchInfo",
    "JBDetectionResult",
    "PageResult",
    # Pipeline
    "TextDetector",
    "DetectorError",
    "PatternMatcher",
    "TagMatcher",
    "JBDetectionPipeline",
    # Unified PDF
    "PdfTypeDetector",
    "PdfType",
    "DigitalTextExtractor",
    "UnifiedPdfProcessor",
    # Visualization
    "PDFAnnotator",
    # Facade (optional)
    "ExcelExporter",
    "TagJBExtractor",
    "DataAnalysis",
]

__version__ = "1.0.0"
