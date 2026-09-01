"""JBDetection — PaddleOCR detector wrapper.

This module is INDEPENDENT of OCR Studio. It does not import from
``ocr_core`` — it wraps ``paddleocr.PaddleOCR`` directly. The wrapping
pattern (lazy init + singleton + thread lock) is inspired by OCR
Studio's ``paddle_engine.py`` but re-implemented from scratch.

Raw PaddleOCR 2.10.0 output shape (verified):

    result = [
        [                                  # result[0] = page 0
            [
                [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],   # quad (4 points)
                ("detected text", 0.97),              # (text, confidence)
            ],
            ...
        ]
    ]

For image input, ``len(result) == 1``. PaddleOCR returns ``None``
instead of an empty list when no text is detected on a page.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, List, Optional

import numpy as np

from .config import Config
from .models import OcrDetection

logger = logging.getLogger("jb_detection.detector")


# ── Exceptions ──────────────────────────────────────────────────────────
class DetectorError(RuntimeError):
    """Raised when the PaddleOCR backend is unavailable or fails."""


# ── TextDetector ────────────────────────────────────────────────────────
class TextDetector:
    """Singleton-style wrapper around ``paddleocr.PaddleOCR``.

    Design
    ------
    - The heavy ``PaddleOCR`` instance (~5 seconds to init) is created
      lazily on the first ``detect`` call. This keeps module import
      cheap and lets tests construct the wrapper without forcing the
      slow init.
    - A process-wide lock guards the init and the per-call ``ocr.ocr``
      invocation. PaddleOCR is **not** thread-safe — concurrent calls
      will segfault. The lock forces serial access.
    - A single instance is shared across all callers via the
      :meth:`get_instance` classmethod. This avoids re-init when
      multiple pipeline stages need OCR.
    """

    _instance: Optional["TextDetector"] = None
    _instance_lock = threading.Lock()

    def __init__(self, config: Optional[Config] = None) -> None:
        if config is None:
            from .config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG
        self._config = config
        self._ocr: Optional[Any] = None
        self._init_lock = threading.Lock()
        self._call_lock = threading.Lock()
        # Stats — exposed for the facade
        self.calls: int = 0
        self.detections: int = 0
        self.errors: int = 0

    # ── Singleton accessor ─────────────────────────────────────────
    @classmethod
    def get_instance(cls, config: Optional[Config] = None) -> "TextDetector":
        """Return the process-wide singleton.

        The first caller wins the config — subsequent callers get the
        same instance even if they pass a different config. This is
        intentional: re-initialising PaddleOCR mid-run is expensive
        and rarely needed.
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(config=config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Drop the singleton (used by tests)."""
        with cls._instance_lock:
            cls._instance = None

    # ── Lazy init ──────────────────────────────────────────────────
    def _ensure_initialized(self) -> None:
        if self._ocr is not None:
            return
        with self._init_lock:
            if self._ocr is not None:
                return
            try:
                from paddleocr import PaddleOCR  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise DetectorError(
                    f"Failed to import paddleocr.PaddleOCR: {exc}. "
                    f"Install with: pip install paddleocr paddlepaddle"
                ) from exc

            logger.info(
                "Initializing PaddleOCR(use_angle_cls=%s, lang=%s, "
                "show_log=%s, use_gpu=%s)",
                self._config.paddle_use_angle_cls, self._config.paddle_lang,
                self._config.paddle_show_log, self._config.paddle_use_gpu,
            )
            try:
                self._ocr = PaddleOCR(
                    use_angle_cls=self._config.paddle_use_angle_cls,
                    lang=self._config.paddle_lang,
                    show_log=self._config.paddle_show_log,
                    use_gpu=self._config.paddle_use_gpu,
                )
            except Exception as exc:
                raise DetectorError(
                    f"PaddleOCR initialization failed: {exc}"
                ) from exc
            logger.info("PaddleOCR instance ready.")

    # ── Public API ─────────────────────────────────────────────────
    def detect(self, image: np.ndarray) -> List[OcrDetection]:
        """Run OCR on a single image.

        Parameters
        ----------
        image:
            BGR numpy array (HxWx3, uint8). PaddleOCR expects BGR.

        Returns
        -------
        List[OcrDetection]
            One entry per detected text region. Sorted in reading order
            (top-to-bottom, left-to-right using the bbox top-left).
            Returns ``[]`` when PaddleOCR finds nothing.
        """
        if image is None or image.size == 0:
            return []

        self._ensure_initialized()
        assert self._ocr is not None  # for type checkers

        # PaddleOCR is NOT thread-safe — guard the call.
        with self._call_lock:
            try:
                raw_result = self._ocr.ocr(image, cls=True)
                self.calls += 1
            except Exception as exc:
                self.errors += 1
                logger.error("PaddleOCR.ocr() raised: %s", exc)
                raise DetectorError(f"PaddleOCR.ocr() raised: {exc}") from exc

        detections = self._parse_raw_result(raw_result)
        self.detections += len(detections)
        return detections

    # ── Parsing ────────────────────────────────────────────────────
    def _parse_raw_result(self, raw_result: Optional[list]) -> List[OcrDetection]:
        """Translate raw PaddleOCR output into :class:`OcrDetection` list.

        PaddleOCR returns ``[page0, page1, ...]`` where each page is
        either ``None`` (no detections) or a list of
        ``[quad, (text, confidence)]`` entries.
        """
        if raw_result is None:
            return []
        if len(raw_result) == 0:
            return []

        page_detections = raw_result[0]
        if page_detections is None:
            return []

        out: List[OcrDetection] = []
        for det in page_detections:
            parsed = self._parse_single(det)
            if parsed is not None:
                out.append(parsed)

        # Deterministic reading order: top→bottom, left→right.
        # PaddleOCR doesn't guarantee any particular order.
        out.sort(key=lambda d: (d.bbox[1], d.bbox[0]))
        return out

    def _parse_single(self, detection: Any) -> Optional[OcrDetection]:
        """Parse one ``[quad, (text, conf)]`` entry."""
        try:
            quad, text_conf = detection
            text, confidence = text_conf
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping malformed PaddleOCR detection %r: %s",
                           detection, exc)
            return None

        # Normalize polygon to list[list[float]]
        try:
            polygon = [[float(p[0]), float(p[1])] for p in quad]
        except (TypeError, ValueError):
            logger.warning("Bad polygon in detection: %r", quad)
            return None

        if len(polygon) != 4:
            logger.warning("Polygon has %d points (expected 4): %r",
                           len(polygon), polygon)
            return None

        # Defensive: text could be None on rare edge cases.
        if text is None:
            text = ""
        text = str(text)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        # Confidence floor
        if confidence < self._config.paddle_min_confidence:
            return None

        # Axis-aligned bbox from quad
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x = int(min(xs))
        y = int(min(ys))
        w = max(1, int(max(xs) - min(xs)))
        h = max(1, int(max(ys) - min(ys)))

        return OcrDetection(
            text=text,
            confidence=confidence,
            polygon=polygon,
            bbox=(x, y, w, h),
        )

    # ── Stats ──────────────────────────────────────────────────────
    def stats(self) -> dict:
        return {
            "calls": self.calls,
            "detections": self.detections,
            "errors": self.errors,
            "initialized": self._ocr is not None,
        }


__all__ = ["TextDetector", "DetectorError"]
