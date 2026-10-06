"""JBDetection — image preprocessor.

This module is the *only* place in the package that knows how to:

1. Load an image file (PNG/JPG/TIFF/BMP) into a BGR numpy array.
2. Render a single PDF page to a BGR numpy array using PyMuPDF.
3. Apply the **unified** preprocessing pipeline (no table/diagram split).

Design notes
------------
PaddleOCR 2.10.0 expects a **BGR** numpy array (it was originally
designed around ``cv2.imread`` which returns BGR). The previous
JBDetection code converted to RGB and then back — a needless round
trip. We hand BGR straight to PaddleOCR.

PyMuPDF (fitz) is used for PDF rendering because:

- It is faster than pdf2image + poppler for large PDFs.
- It is pure-Python installable (no system poppler dependency).
- It supports per-page rendering via ``page.get_pixmap(matrix=...)``,
  which matches the lazy generator pattern used by OCR Studio's
  ``pdf_reader.py``.

For very large PDFs we auto-reduce the DPI to avoid OOM — same strategy
as OCR Studio.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Generator, Iterator, Optional, Tuple

import cv2
import numpy as np

from .config import Config

logger = logging.getLogger("jb_detection.image_preprocessor")


# ── PyMuPDF colorspace compatibility shim ───────────────────────────────
# PyMuPDF renamed the gray colorspace constant across versions. We try
# the new name first and fall back gracefully.
def _get_gray_colorspace():
    try:
        import fitz  # type: ignore
    except Exception:
        return None
    for attr in ("cs_GRAY", "csGRAY", "COLORSPACE_GRAY"):
        if hasattr(fitz, attr):
            return getattr(fitz, attr)
    return None


_CS_GRAY = None  # resolved lazily on first PDF render


# ── Image loading ───────────────────────────────────────────────────────
def load_image(path: os.PathLike) -> np.ndarray:
    """Load an image file into a BGR numpy array.

    Parameters
    ----------
    path:
        Path to a PNG/JPG/JPEG/BMP/TIFF file.

    Returns
    -------
    np.ndarray
        BGR uint8 array (HxWx3). Single-channel images are promoted to
        3-channel BGR.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file exists but cannot be decoded as an image.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file does not exist: {p}")

    # Read via imdecode (handles non-ASCII paths better than imread).
    try:
        with open(p, "rb") as fh:
            buffer = np.frombuffer(fh.read(), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Failed to read {p}: {exc}") from exc

    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"OpenCV could not decode image: {p}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


# ── PDF rendering ───────────────────────────────────────────────────────
def load_pdf_page(page: "object", dpi: int = 300) -> np.ndarray:
    """Render a single PyMuPDF ``page`` to a BGR numpy array.

    Parameters
    ----------
    page:
        A ``fitz.Page`` instance.
    dpi:
        Render DPI. 72 DPI = original PDF point size; 300 DPI is the
        standard for OCR-quality rendering.

    Returns
    -------
    np.ndarray
        BGR uint8 array.
    """
    global _CS_GRAY
    if _CS_GRAY is None:
        _CS_GRAY = _get_gray_colorspace()

    import fitz  # local import; keeps top-level cheap

    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=matrix)

    # Build numpy view of the raw samples
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    if pix.n == 1:
        arr = arr.reshape(pix.height, pix.width)
        img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif pix.n == 3:
        arr = arr.reshape(pix.height, pix.width, 3)
        # PyMuPDF gives RGB; convert to BGR for cv2/PaddleOCR.
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif pix.n == 4:
        arr = arr.reshape(pix.height, pix.width, 4)
        img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        # Fallback: try PIL
        from PIL import Image  # type: ignore
        img = np.array(Image.frombytes("RGBA" if pix.alpha else "RGB",
                                       (pix.width, pix.height),
                                       pix.samples))
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def _compute_effective_dpi(config: Config, page_count: int) -> int:
    """Auto-reduce DPI for large PDFs — same strategy as OCR Studio."""
    if not config.pdf_auto_dpi_enabled or page_count == 0:
        return config.pdf_dpi
    if page_count > config.pdf_auto_dpi_threshold_2:
        logger.warning(
            "Large PDF (%d pages > %d): reducing DPI %d → %d to prevent OOM",
            page_count, config.pdf_auto_dpi_threshold_2,
            config.pdf_dpi, config.pdf_auto_dpi_lower,
        )
        return config.pdf_auto_dpi_lower
    if page_count > config.pdf_auto_dpi_threshold_1:
        logger.warning(
            "Large PDF (%d pages > %d): reducing DPI %d → %d to prevent OOM",
            page_count, config.pdf_auto_dpi_threshold_1,
            config.pdf_dpi, config.pdf_auto_dpi_low,
        )
        return config.pdf_auto_dpi_low
    return config.pdf_dpi


def render_pdf_to_images(pdf_path: os.PathLike,
                          config: Optional[Config] = None,
                          max_pages: Optional[int] = None,
                          ) -> Generator[Tuple[int, np.ndarray, int, int], None, None]:
    """Lazy generator: yield ``(page_number, image_bgr, width, height)``.

    This is the JBDetection equivalent of OCR Studio's
    ``PdfReader.render_pages_lazy``. Pages are rendered ONE AT A TIME so
    peak memory is ~25MB instead of ~7.5GB for a 300-page PDF.

    Parameters
    ----------
    pdf_path:
        Path to a PDF file.
    config:
        :class:`Config` instance. Defaults to :data:`DEFAULT_CONFIG`.
    max_pages:
        Optional override for ``config.pdf_max_pages``.

    Yields
    ------
    (page_number, image_bgr, width, height)
        ``page_number`` is 1-indexed.
    """
    if config is None:
        from .config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG

    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"PDF file does not exist: {p}")

    size = p.stat().st_size
    if size > config.pdf_max_file_size_mb * 1024 * 1024:
        raise ValueError(
            f"PDF too large: {p.name} = {size / 1024 / 1024:.1f} MB "
            f"(limit {config.pdf_max_file_size_mb} MB)"
        )

    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"PyMuPDF (fitz) not available: {exc}. "
            f"Install with: pip install PyMuPDF"
        ) from exc

    doc = fitz.open(str(p))
    try:
        page_count = len(doc)
        cap = max_pages if max_pages is not None else config.pdf_max_pages
        effective_dpi = _compute_effective_dpi(config, page_count)
        total = min(page_count, cap)

        logger.info(
            "Rendering PDF: %s (pages=%d, dpi=%d→%d)",
            p.name, page_count, config.pdf_dpi, effective_dpi,
        )

        for idx in range(total):
            try:
                page = doc.load_page(idx)
                img = load_pdf_page(page, dpi=effective_dpi)
                h, w = img.shape[:2]
                yield (idx + 1, img, w, h)
            except Exception as exc:
                logger.error("Failed to render page %d: %s", idx + 1, exc)
                continue
            finally:
                # Periodic GC for very large PDFs
                if (idx + 1) % config.gc_interval_pages == 0:
                    gc.collect()
    finally:
        try:
            doc.close()
        except Exception:
            pass


def _normalize_highlights(image: np.ndarray) -> np.ndarray:
    """Whiten bright yellow highlighter while retaining dark/colored ink."""
    if image.ndim != 3 or image.shape[2] != 3:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    highlight = cv2.inRange(hsv, (15, 35, 170), (42, 255, 255))
    if not cv2.countNonZero(highlight):
        return image
    cleaned = image.copy()
    cleaned[highlight != 0] = 255
    return cleaned


def _remove_ruling_lines(gray: np.ndarray) -> np.ndarray:
    """Remove long, thin table rules while protecting crossing ink strokes.

    Geometry is unchanged. Only long axis-aligned strokes are considered;
    short glyph bars and broad filled objects are left alone.
    """
    height, width = gray.shape
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = np.zeros_like(gray)
    max_thickness = max(3, min(12, min(height, width) // 150))
    for horizontal in (True, False):
        length = max(80, (width if horizontal else height) // 30)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1) if horizontal else (1, length))
        opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        valid = np.zeros(count, dtype=np.uint8)
        thicknesses = []
        for index in range(1, count):
            _, _, w, h, _ = stats[index]
            thickness, span = (h, w) if horizontal else (w, h)
            if thickness <= max_thickness and span >= length:
                valid[index] = 255
                thicknesses.append(thickness)
        if not thicknesses:
            continue
        lines = valid[labels]
        distance = int(np.median(thicknesses)) + 2
        before, after = np.zeros_like(ink), np.zeros_like(ink)
        if horizontal and distance < height:
            before[distance:] = ink[:-distance]
            after[:-distance] = ink[distance:]
            guard_kernel = np.ones((1, 3), np.uint8)
        elif not horizontal and distance < width:
            before[:, distance:] = ink[:, :-distance]
            after[:, :-distance] = ink[:, distance:]
            guard_kernel = np.ones((3, 1), np.uint8)
        else:
            continue
        crossing = cv2.dilate(cv2.bitwise_and(before, after), guard_kernel)
        removable = cv2.bitwise_and(lines, cv2.bitwise_not(crossing))
        mask = cv2.bitwise_or(mask, removable)
    if not cv2.countNonZero(mask):
        return gray
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)


# ── Preprocessing (UNIFIED pipeline) ────────────────────────────────────
def preprocess(image: np.ndarray,
               upscale: int = 1,
               config: Optional[Config] = None,
               ) -> np.ndarray:
    """Unified preprocessing for both diagrams and table-style PDFs.

    Steps
    -----
    1. Optional upscaling (for low-DPI scans).
    2. Neutralize bright yellow highlights, then convert to grayscale.
       Remove long thin ruling lines with protection for crossing text strokes.
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization) —
       dramatically improves OCR on low-contrast scans.
    4. Light Gaussian blur (removes 1-pixel noise without destroying
       thin strokes).
    5. Otsu threshold (binarises the image; PaddleOCR works well on
       either binary or grayscale, so we return a 3-channel BGR image
       so callers don't need to special-case).

    The previous code split this into "diagram" and "table" branches.
    Empirically, the same CLAHE + Otsu pipeline works for both — the
    split was a historical artefact of the Tesseract-era code.

    Parameters
    ----------
    image:
        BGR numpy array.
    upscale:
        Integer upscale factor (1 = no scaling, 2 = double size).
    config:
        :class:`Config` instance. Defaults to :data:`DEFAULT_CONFIG`.

    Returns
    -------
    np.ndarray
        Preprocessed BGR uint8 array (same dtype as input).
    """
    if config is None:
        from .config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG

    if image is None or image.size == 0:
        raise ValueError("preprocess: empty image")

    # 1. Upscale
    if upscale and upscale > 1:
        h, w = image.shape[:2]
        image = cv2.resize(image, (w * upscale, h * upscale),
                            interpolation=cv2.INTER_CUBIC)

    # 2. Remove colored background before global thresholding.
    if config.preprocess_remove_highlights:
        image = _normalize_highlights(image)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if config.preprocess_remove_ruling_lines:
        gray = _remove_ruling_lines(gray)

    # 3. CLAHE — adaptive contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=float(config.preprocess_clahe_clip),
        tileGridSize=(int(config.preprocess_clahe_tile),
                      int(config.preprocess_clahe_tile)),
    )
    gray = clahe.apply(gray)

    # 4. Light Gaussian blur
    k = int(config.preprocess_gaussian_kernel)
    if k > 0:
        # kernel size must be odd and positive
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # 5. Otsu threshold
    if config.preprocess_use_otsu:
        _, gray = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Back to 3-channel BGR for PaddleOCR (which expects BGR).
    if gray.ndim == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray


__all__ = [
    "load_image",
    "load_pdf_page",
    "render_pdf_to_images",
    "preprocess",
]
