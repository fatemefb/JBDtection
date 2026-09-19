"""JBDetection — OCR GPU smoke test.

Verifies that PaddleOCR can be initialised and can run real OCR
inference. Uses the GPU path (PaddleOCR 2.10.0 `use_gpu=True`).

Run as a script:

    python -m tests.test_ocr_gpu

Exit codes:
    0 = OCR initialization + inference succeeded
    1 = Failure (with diagnostic output)
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path


# Sample image with known text content. We generate it at runtime so
# the test is self-contained (no external fixtures needed).
SAMPLE_TEXTS = ["JB-101", "TE-5223", "PT-1014", "MC-200", "SPARE"]


def _make_test_image():
    """Generate a BGR image with the SAMPLE_TEXTS drawn on it."""
    import cv2
    import numpy as np

    img = np.ones((400, 800, 3), dtype=np.uint8) * 255
    for i, text in enumerate(SAMPLE_TEXTS):
        y = 60 + i * 60
        cv2.putText(img, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def main() -> int:
    # Preload libz
    try:
        import ctypes
        ctypes.CDLL("libz.so.1", mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

    print("=" * 60)
    print("JBDetection — OCR GPU Smoke Test")
    print("=" * 60)
    print()

    # ── 1. Validate GPU environment FIRST ─────────────────────────────
    print("[1/5] Validating GPU environment...")
    try:
        from jb_detection.gpu_validation import (
            validate_gpu_environment, GPUEnvironmentError,
        )
        env = validate_gpu_environment()
        print(f"      PaddlePaddle: {env.paddle_version}")
        print(f"      PaddleOCR:   {env.paddleocr_version}")
        print(f"      Device:      {env.device}")
        print(f"      GPU:         {env.gpu_name or '(unknown)'}")
        print(f"      GPU count:   {env.gpu_count}")
        print(f"      CUDA build:  {env.compiled_with_cuda}")
        print(f"      Tensor op:  {'OK' if env.tensor_op_ok else 'FAILED'}")
        if not env.ok:
            print(f"      FAIL: GPU validation failed: {env.errors}")
            return 1
    except GPUEnvironmentError as exc:
        print(f"      FAIL: GPUEnvironmentError raised:")
        print(str(exc))
        return 1
    except Exception as exc:
        print(f"      FAIL: unexpected error: {exc}")
        traceback.print_exc()
        return 1

    # ── 2. Initialize PaddleOCR with use_gpu=True ─────────────────────
    print()
    print("[2/5] Initializing PaddleOCR (use_gpu=True)...")
    try:
        from paddleocr import PaddleOCR

        # PaddleOCR 2.10.0 API: use_gpu is a supported kwarg (default True).
        # We pass it explicitly for clarity.
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False,
            use_gpu=True,
        )
        print(f"      PaddleOCR instance: {type(ocr).__name__}")

        # Verify PaddleOCR did NOT silently fall back to CPU
        actual_use_gpu = getattr(ocr, "params", None)
        if actual_use_gpu is not None:
            actual_flag = getattr(actual_use_gpu, "use_gpu", None)
            print(f"      PaddleOCR params.use_gpu: {actual_flag}")
            if actual_flag is False:
                print("      FAIL: PaddleOCR silently fell back to CPU.")
                print("      This means the GPU is not actually being used.")
                return 1
    except Exception as exc:
        print(f"      FAIL: PaddleOCR initialization raised: {exc}")
        traceback.print_exc()
        return 1

    # ── 3. Generate a test image ──────────────────────────────────────
    print()
    print("[3/5] Generating test image with known text...")
    try:
        img = _make_test_image()
        print(f"      Image shape: {img.shape} (BGR uint8)")
    except Exception as exc:
        print(f"      FAIL: could not generate test image: {exc}")
        return 1

    # ── 4. Run OCR inference ──────────────────────────────────────────
    print()
    print("[4/5] Running PaddleOCR inference on test image...")
    try:
        result = ocr.ocr(img, cls=True)
        if result is None:
            print("      FAIL: PaddleOCR returned None.")
            return 1
        if not result or not result[0]:
            print("      FAIL: PaddleOCR returned no detections.")
            print("      (The test image has clear text — this should not happen.)")
            return 1

        detections = result[0]
        print(f"      Detections: {len(detections)}")
        print()
        print("      Detected text regions:")
        for i, det in enumerate(detections):
            text = det[1][0]
            conf = det[1][1]
            print(f"        [{i}] text={text!r:30s}  conf={conf:.4f}")
    except Exception as exc:
        print(f"      FAIL: PaddleOCR inference raised: {exc}")
        traceback.print_exc()
        return 1

    # ── 5. Verify expected texts were found ───────────────────────────
    print()
    print("[5/5] Verifying detected text matches expected content...")
    detected_texts = " ".join(d[1][0] for d in detections).upper()
    found = 0
    for expected in SAMPLE_TEXTS:
        if expected.upper() in detected_texts:
            found += 1
            print(f"      ✓ Found: {expected}")
        else:
            print(f"      ✗ Missing: {expected}")

    if found == 0:
        print()
        print("      FAIL: None of the expected texts were detected.")
        return 1

    print()
    print(f"      Found {found}/{len(SAMPLE_TEXTS)} expected texts.")
    if found < len(SAMPLE_TEXTS):
        print("      (Partial match — OCR is working but some texts were missed.)")

    # ── Summary ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("OCR GPU SMOKE TEST: PASS")
    print(f"  PaddlePaddle:  {env.paddle_version}")
    print(f"  PaddleOCR:     {env.paddleocr_version}")
    print(f"  Device:        {env.device}")
    print(f"  GPU:           {env.gpu_name or '(unknown)'}")
    print(f"  Detections:    {len(detections)}")
    print(f"  Texts found:   {found}/{len(SAMPLE_TEXTS)}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
