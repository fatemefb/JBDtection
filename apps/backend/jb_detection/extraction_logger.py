"""JBDetection — Extraction Logger.

Centralised, structured logging for the extraction pipeline. Captures
every stage from raw OCR/digital-text detection through pattern
classification to tag matching, so the entire decision trail for any
given detection is reproducible and inspectable.

The logger is designed to be:
  • **Non-intrusive** — emits structured records via Python ``logging``
    so it integrates with any existing log infrastructure.
  • **Granular** — one log record per detection decision (not per page)
    so you can trace exactly why a token was classified as JB / MC /
    Tag / Cable / SPARE / Unknown.
  • **Future-proof** — the structured ``DetectionRecord`` schema is
    LLM-friendly: a downstream LLM agent can consume the JSONL output
    and learn / propose new patterns.

Stages logged
-------------
1. ``raw_extraction``     — every OcrDetection produced by the
                            DigitalTextExtractor or PaddleOCR.
2. ``pre_classification`` — tokens that survived stop-word + length
                            filtering and are about to be classified.
3. ``classification``     — the final category assigned (JB / MC /
                            Tag / Cable / SPARE / Unknown) with the
                            reason + which regex fired.
4. ``tag_match``          — for tags: the match_type (exact / similar
                            / unmatched) and score against the IO List.
5. ``final_summary``      — page-level summary counts.

Usage
-----
The logger is a process-wide singleton::

    from jb_detection.extraction_logger import log_extraction
    log_extraction("raw_extraction", page=1, text="FIT-100-14",
                   bbox=(765, 405, 106, 24), confidence=1.0,
                   source="digital")

By default, logs go to the ``jb_detection.extraction`` logger, which
the application configures. To dump to a JSONL file::

    from jb_detection.extraction_logger import enable_jsonl_log
    enable_jsonl_log("/tmp/jbdetection_extraction.jsonl")

JSONL schema (one record per line)
----------------------------------
    {
      "stage": "raw_extraction",
      "ts": "2026-09-07T05:43:15.123",
      "page": 1,
      "source": "digital",          // or "paddleocr"
      "text": "FIT-100-14",
      "bbox": [765, 405, 106, 24],
      "confidence": 1.0,
      "category": null,             // filled in at classification
      "reason": null
    }
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Dedicated logger — application code can attach handlers to it.
logger = logging.getLogger("jb_detection.extraction")
logger.setLevel(logging.DEBUG)


# ── JSONL file sink (optional) ─────────────────────────────────────────
_jsonl_path: Optional[str] = None
_jsonl_lock = threading.Lock()
_jsonl_file = None


def enable_jsonl_log(path: str) -> None:
    """Enable writing every extraction record to a JSONL file.

    The file is opened in append mode and flushed after each record
    so partial logs survive crashes. Call :func:`disable_jsonl_log`
    to close the file.
    """
    global _jsonl_path, _jsonl_file
    with _jsonl_lock:
        if _jsonl_file is not None:
            try:
                _jsonl_file.close()
            except Exception:
                pass
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        _jsonl_path = path
        _jsonl_file = open(path, "a", encoding="utf-8", buffering=1)
        logger.info("Extraction JSONL logging enabled: %s", path)


def disable_jsonl_log() -> None:
    """Close the JSONL log file if open."""
    global _jsonl_path, _jsonl_file
    with _jsonl_lock:
        if _jsonl_file is not None:
            try:
                _jsonl_file.close()
            except Exception:
                pass
        _jsonl_file = None
        _jsonl_path = None


# ── Structured record ──────────────────────────────────────────────────
@dataclass
class DetectionRecord:
    """One structured log record for a single detection decision.

    Fields are kept flat (no nested dicts) so JSONL consumption by an
    LLM agent is straightforward.
    """
    stage: str                    # raw_extraction | pre_classification | classification | tag_match | final_summary
    ts: str = ""                  # ISO 8601 timestamp
    page: int = 0                 # 1-indexed page number
    source: str = ""             # "digital" | "paddleocr" | "manual"
    text: str = ""                # the raw OCR / extracted text
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0
    category: Optional[str] = None    # JB | MC | Tag | Cable | SPARE | Unknown
    reason: Optional[str] = None      # human-readable reason for the decision
    pattern_name: Optional[str] = None  # which regex/pattern fired
    pattern_match: Optional[str] = None  # the actual matched substring
    extras: Dict[str, Any] = field(default_factory=dict)  # free-form

    def __post_init__(self):
        if not self.ts:
            self.ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # bbox as list (JSON-friendly)
        d["bbox"] = list(self.bbox)
        return d


# ── Public API ─────────────────────────────────────────────────────────
def log_extraction(
    stage: str,
    *,
    page: int = 0,
    source: str = "",
    text: str = "",
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0),
    confidence: float = 0.0,
    category: Optional[str] = None,
    reason: Optional[str] = None,
    pattern_name: Optional[str] = None,
    pattern_match: Optional[str] = None,
    **extras,
) -> None:
    """Emit one structured extraction record.

    Parameters
    ----------
    stage:
        One of ``raw_extraction``, ``pre_classification``,
        ``classification``, ``tag_match``, ``final_summary``.
    page:
        1-indexed page number.
    source:
        ``"digital"`` for native PDF extraction, ``"paddleocr"`` for OCR.
    text:
        The raw text of the detection.
    bbox:
        ``(x, y, width, height)`` in pixel coordinates.
    confidence:
        OCR confidence (1.0 for digital extraction).
    category:
        Final classification — only set at the ``classification`` stage.
    reason:
        Short human-readable explanation of the decision.
    pattern_name:
        Name of the regex / pattern that fired (e.g. ``"TAG_PATTERN"``,
        ``"CABLE_PATTERN"``, ``"JB_PATTERN"``).
    pattern_match:
        The substring matched by the pattern.
    **extras:
        Any additional fields to include in the record.
    """
    record = DetectionRecord(
        stage=stage,
        page=page,
        source=source,
        text=text,
        bbox=bbox,
        confidence=confidence,
        category=category,
        reason=reason,
        pattern_name=pattern_name,
        pattern_match=pattern_match,
        extras=extras,
    )

    # Emit to the standard Python logger (DEBUG level by default).
    logger.debug("%s | page=%d source=%s text=%r category=%s reason=%s",
                stage, page, source, text, category, reason)

    # Emit to JSONL file if enabled.
    if _jsonl_file is not None:
        with _jsonl_lock:
            try:
                _jsonl_file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            except Exception:
                pass  # never let logging crash the pipeline


def log_raw_extraction(
    page: int,
    source: str,
    detections: List[Any],
) -> None:
    """Convenience: log every detection in a page's raw extraction."""
    for det in detections:
        log_extraction(
            "raw_extraction",
            page=page,
            source=source,
            text=det.text,
            bbox=det.bbox,
            confidence=det.confidence,
        )


def log_final_summary(
    page: int,
    source: str,
    counts: Dict[str, int],
    total_detections: int,
) -> None:
    """Convenience: log the page-level summary."""
    log_extraction(
        "final_summary",
        page=page,
        source=source,
        text=f"page_summary",
        extras={
            "counts": counts,
            "total_detections": total_detections,
        },
    )


__all__ = [
    "DetectionRecord",
    "log_extraction",
    "log_raw_extraction",
    "log_final_summary",
    "enable_jsonl_log",
    "disable_jsonl_log",
]
