"""Read the normal image first, then recover missing JBs from a line-free view."""
from __future__ import annotations

from dataclasses import replace
import logging
from typing import List

import numpy as np

from .config import Config
from .image_preprocessor import preprocess
from .models import OcrDetection

logger = logging.getLogger("jb_detection.ocr_recovery")


def detect_with_jb_recovery(image, detector, matcher, config: Config) -> List[OcrDetection]:
    try:
        primary = preprocess(image, config=config)
    except Exception as exc:
        logger.warning("Preprocessing failed; using original image: %s", exc)
        primary = image
    detections = detector.detect(primary)
    if (not config.ocr_recover_missing_jb
            or (matcher._explicit_patterns["jb"] and not matcher.jb_examples_list)
            or any(matcher._is_jb_token(item.text) for item in detections)):
        return detections

    # Line deletion is intentionally a recovery view: preserve every original
    # detection, and add only JB identifiers validated by configured patterns.
    try:
        alternate_config = replace(config, preprocess_remove_ruling_lines=not config.preprocess_remove_ruling_lines)
        alternate = preprocess(image, config=alternate_config)
        if np.array_equal(primary, alternate):
            return detections
        recovered = [item for item in detector.detect(alternate) if matcher._is_jb_token(item.text)]
    except Exception as exc:
        logger.warning("Missing-JB recovery failed; keeping primary detections: %s", exc)
        return detections
    if recovered:
        logger.info("Recovered %d JB occurrence(s) from the alternate OCR view", len(recovered))
    return list(detections) + recovered
