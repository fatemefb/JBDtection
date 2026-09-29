"""JBDetection — domain models.

Pure data containers (dataclasses) used by every module in the package.
Importing this module has zero third-party dependencies beyond the
standard library — it is safe to import in test environments where
numpy / cv2 / paddleocr are not installed.

Coordinate convention
---------------------
All coordinates in :class:`OcrDetection` and downstream containers are
*pixel* coordinates relative to the rasterized page image (after
preprocessing). They are NOT PDF-point coordinates. To convert back to
PDF points:

    pt = px * 72 / dpi

where ``dpi`` is the rendering DPI recorded on the page.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# Type aliases — kept as ``Any`` so that numpy isn't required to import
# this module. Real users will see ``np.ndarray``.
BBox = Tuple[int, int, int, int]   # (x, y, width, height)
Polygon = List[List[float]]        # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]


# ── OCR detection ───────────────────────────────────────────────────────
@dataclass
class OcrDetection:
    """A single text region detected by PaddleOCR.

    Attributes
    ----------
    text:
        Recognized text (already stripped, original case preserved).
    confidence:
        Float in ``[0.0, 1.0]`` — PaddleOCR's confidence score.
    polygon:
        4-point polygon in pixel coordinates (top-left, top-right,
        bottom-right, bottom-left — but PaddleOCR doesn't guarantee
        ordering, so consumers should use :attr:`bbox` for axis-aligned
        work).
    bbox:
        Axis-aligned bounding box ``(x, y, width, height)`` derived from
        :attr:`polygon`.
    """

    text: str
    confidence: float
    polygon: Polygon
    bbox: BBox

    # ── Convenience helpers ────────────────────────────────────────
    @property
    def x(self) -> int:
        return int(self.bbox[0])

    @property
    def y(self) -> int:
        return int(self.bbox[1])

    @property
    def width(self) -> int:
        return int(self.bbox[2])

    @property
    def height(self) -> int:
        return int(self.bbox[3])

    @property
    def center(self) -> Tuple[float, float]:
        """Center point ``(cx, cy)`` — used for reading-order sorting."""
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def __post_init__(self) -> None:
        # Defensive: ensure types are stable even when constructed from
        # raw PaddleOCR output (which uses floats / numpy scalars).
        if not isinstance(self.text, str):
            self.text = "" if self.text is None else str(self.text)
        self.confidence = float(self.confidence)
        if not isinstance(self.polygon, list):
            self.polygon = [[float(p[0]), float(p[1])] for p in self.polygon]
        self.bbox = (int(self.bbox[0]), int(self.bbox[1]),
                     int(self.bbox[2]), int(self.bbox[3]))


# ── Tag match info ──────────────────────────────────────────────────────
@dataclass
class TagMatchInfo:
    """Detailed match result for a single OCR-detected tag.

    Attributes
    ----------
    match_type:
        One of ``"exact"``, ``"similar"``, ``"unmatched"``,
        ``"unmatched_candidate"``.
    score:
        Similarity score in ``[0.0, 1.0]``. ``1.0`` for exact matches.
    ocr_text:
        The OCR text that was matched (preserves original case).
    matched_tag:
        The IO List tag that was matched (or ``""`` if unmatched).
    bbox:
        Bounding box of the OCR detection, in pixel coordinates.
    reason:
        Optional human-readable explanation (e.g.
        ``"OCR confusion: 'O' → '0'"``).
    """

    match_type: str = "unmatched"
    score: float = 0.0
    ocr_text: str = ""
    matched_tag: str = ""
    bbox: BBox = (0, 0, 0, 0)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_type": self.match_type,
            "score": float(self.score),
            "ocr_text": self.ocr_text,
            "matched_tag": self.matched_tag,
            "bbox": {
                "x": int(self.bbox[0]),
                "y": int(self.bbox[1]),
                "width": int(self.bbox[2]),
                "height": int(self.bbox[3]),
            },
            "reason": self.reason,
        }


# ── Page result (the 9-tuple as a structured object) ───────────────────
@dataclass
class JBDetectionResult:
    """Result of running the pipeline on a single page.

    Field order matches the original 9-tuple returned by
    ``extract_from_image`` so that the public API stays backward
    compatible:

    1. tags
    2. jb_identifiers
    3. mc_identifiers
    4. cable_descriptions
    5. spare_identifiers
    6. tag_to_number
    7. raw_cable_descriptions
    8. tag_match_info
    9. all_ocr_tags
    """

    tags: Set[str] = field(default_factory=set)
    jb_identifiers: Set[str] = field(default_factory=set)
    mc_identifiers: Set[str] = field(default_factory=set)
    cable_descriptions: List[str] = field(default_factory=list)
    spare_identifiers: List[str] = field(default_factory=list)
    tag_to_number: Dict[str, int] = field(default_factory=dict)
    raw_cable_descriptions: List[str] = field(default_factory=list)
    tag_match_info: Dict[str, TagMatchInfo] = field(default_factory=dict)
    all_ocr_tags: Set[str] = field(default_factory=set)

    # Convenience: positions per tag, kept separately so the structured
    # tuple stays backward compatible. Populated by PatternMatcher.
    tag_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    spare_positions: List[Dict[str, Any]] = field(default_factory=list)

    def to_tuple(self) -> Tuple[Any, ...]:
        """Return the legacy 9-tuple form for backward compatibility."""
        # tag_match_info must be exported as Dict[str, Dict] (the original
        # format) so existing consumers that do info["match_type"] keep
        # working.
        tmi_dict = {k: v.to_dict() if isinstance(v, TagMatchInfo) else v
                    for k, v in self.tag_match_info.items()}
        return (
            set(self.tags),
            set(self.jb_identifiers),
            set(self.mc_identifiers),
            list(self.cable_descriptions),
            list(self.spare_identifiers),
            dict(self.tag_to_number),
            list(self.raw_cable_descriptions),
            tmi_dict,
            set(self.all_ocr_tags),
        )

    @classmethod
    def from_tuple(cls, tpl: Tuple[Any, ...]) -> "JBDetectionResult":
        """Build a :class:`JBDetectionResult` from a legacy 9-tuple."""
        if not isinstance(tpl, (tuple, list)):
            return cls()
        # Pad to 9 elements
        padded = list(tpl) + [None] * max(0, 9 - len(tpl))
        tags, jbs, mcs, cables, spares, t2n, raw_cables, tmi, ocr_tags = padded[:9]
        # Normalise tag_match_info — accept either TagMatchInfo or dict
        normalised_tmi: Dict[str, TagMatchInfo] = {}
        if isinstance(tmi, dict):
            for k, v in tmi.items():
                if isinstance(v, TagMatchInfo):
                    normalised_tmi[str(k)] = v
                elif isinstance(v, dict):
                    bbox = v.get("bbox", (0, 0, 0, 0))
                    if isinstance(bbox, dict):
                        bbox = (bbox.get("x", 0), bbox.get("y", 0),
                                bbox.get("width", 0), bbox.get("height", 0))
                    normalised_tmi[str(k)] = TagMatchInfo(
                        match_type=v.get("match_type", "unmatched"),
                        score=float(v.get("score", 0.0) or 0.0),
                        ocr_text=str(v.get("ocr_text", "")),
                        matched_tag=str(v.get("matched_tag", "")),
                        bbox=tuple(bbox) if isinstance(bbox, (list, tuple)) else (0, 0, 0, 0),
                        reason=str(v.get("reason", "")),
                    )
        return cls(
            tags=set(tags) if tags else set(),
            jb_identifiers=set(jbs) if jbs else set(),
            mc_identifiers=set(mcs) if mcs else set(),
            cable_descriptions=list(cables) if cables else [],
            spare_identifiers=list(spares) if spares else [],
            tag_to_number=dict(t2n) if t2n else {},
            raw_cable_descriptions=list(raw_cables) if raw_cables else [],
            tag_match_info=normalised_tmi,
            all_ocr_tags=set(ocr_tags) if ocr_tags else set(),
        )


@dataclass
class PageResult:
    """A :class:`JBDetectionResult` plus its page number and image dims."""

    page_number: int
    result: JBDetectionResult
    width: int = 0
    height: int = 0
    dpi: int = 300

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "result": {
                "tags": sorted(self.result.tags),
                "jb_identifiers": sorted(self.result.jb_identifiers),
                "mc_identifiers": sorted(self.result.mc_identifiers),
                "cable_descriptions": list(self.result.cable_descriptions),
                "spare_identifiers": list(self.result.spare_identifiers),
                "tag_to_number": dict(self.result.tag_to_number),
                "raw_cable_descriptions": list(self.result.raw_cable_descriptions),
                "tag_match_info": {k: v.to_dict() for k, v in self.result.tag_match_info.items()},
                "all_ocr_tags": sorted(self.result.all_ocr_tags),
            },
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


__all__ = [
    "BBox",
    "Polygon",
    "OcrDetection",
    "TagMatchInfo",
    "JBDetectionResult",
    "PageResult",
]
