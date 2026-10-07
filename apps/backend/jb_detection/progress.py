"""Optional, failure-isolated progress events for long PDF processing jobs."""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

ProgressCallback = Callable[[Dict[str, Any]], None]
logger = logging.getLogger(__name__)


def report_progress(callback: Optional[ProgressCallback], stage: str,
                    pdf_path: str, current: int, total: int) -> None:
    if callback is None:
        return
    try:
        callback({"stage": stage, "pdf_path": str(pdf_path),
                  "pdf_name": os.path.basename(pdf_path),
                  "current": current, "total": total})
    except Exception as exc:
        # A status-store failure must not discard a successfully processed page.
        logger.warning("PDF progress update failed: %s", exc)
