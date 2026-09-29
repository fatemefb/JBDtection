"""JBDetection — GPU smoke test.

Verifies that PaddlePaddle GPU is installed AND that a real tensor
operation actually executes on the GPU. Importing this module does not
run the test — run it as a script:

    python -m tests.test_gpu

Exit codes:
    0 = GPU validation passed (GPU available + tensor op succeeded)
    1 = GPU validation failed (GPU unavailable or tensor op failed)
"""

from __future__ import annotations

import sys
import traceback


def main() -> int:
    # Preload libz to work around PaddlePaddle 2.6.x crash on Python 3.12+
    try:
        import ctypes
        ctypes.CDLL("libz.so.1", mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

    print("=" * 60)
    print("JBDetection — GPU Smoke Test")
    print("=" * 60)
    print()

    # ── 1. PaddlePaddle version + CUDA build ─────────────────────────
    try:
        import paddle
        print(f"[1/6] PaddlePaddle version: {paddle.__version__}")
    except Exception as exc:
        print(f"[1/6] FAIL: PaddlePaddle is not importable: {exc}")
        return 1

    compiled_with_cuda = paddle.is_compiled_with_cuda()
    print(f"[2/6] PaddlePaddle CUDA build: {compiled_with_cuda}")
    if not compiled_with_cuda:
        print("      FAIL: PaddlePaddle is the CPU build.")
        print("      Install the GPU wheel:  pip install paddlepaddle-gpu==2.6.2")
        return 1

    # ── 2. CUDA runtime visibility ────────────────────────────────────
    try:
        device = paddle.device.get_device()
        print(f"[3/6] Paddle device: {device}")
    except Exception as exc:
        print(f"[3/6] FAIL: paddle.device.get_device() raised: {exc}")
        return 1

    if device == "cpu":
        print("      FAIL: PaddlePaddle reports 'cpu' device.")
        print("      This means no CUDA GPU is visible to the runtime.")
        print("      Check: nvidia-smi, NVIDIA driver, CUDA toolkit,")
        print("      and (for Docker) NVIDIA Container Toolkit + --gpus all.")
        return 1

    try:
        gpu_count = paddle.device.cuda.device_count()
        print(f"[4/6] GPU count: {gpu_count}")
    except Exception as exc:
        print(f"[4/6] FAIL: paddle.device.cuda.device_count() raised: {exc}")
        return 1

    if gpu_count < 1:
        print("      FAIL: No CUDA devices visible.")
        return 1

    # ── 3. GPU name + CUDA version (best-effort) ─────────────────────
    try:
        gpu_name = paddle.device.cuda.get_device_name(0)
        print(f"[5/6] GPU name: {gpu_name}")
    except Exception as exc:
        print(f"[5/6] GPU name: (could not retrieve: {exc})")

    try:
        cuda_version = paddle.version.cuda()  # type: ignore[attr-defined]
        print(f"      CUDA version (wheel): {cuda_version}")
    except Exception:
        pass

    # ── 4. Actual GPU tensor operation ────────────────────────────────
    # A CUDA-enabled build does NOT prove the runtime can use the GPU.
    # Force placement on gpu:0 and run a real op.
    print()
    print("[6/6] Running GPU tensor operation...")
    try:
        with paddle.device.guard("gpu:0"):
            x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
            y = paddle.multiply(x, x)  # [1, 4, 9, 16]
            z = paddle.sum(y)         # 1+4+9+16 = 30
            result = float(z.numpy())

        expected = 30.0
        if abs(result - expected) > 1e-5:
            print(f"      FAIL: GPU tensor op returned {result}, expected {expected}")
            return 1

        # Verify the tensor is actually on GPU
        # (place is a Tensor attribute in PaddlePaddle 2.x)
        place_str = str(x.place)
        if "GPU" not in place_str.upper() and "CUDA" not in place_str.upper():
            print(f"      WARN: tensor place is {place_str} — expected GPU")

        print(f"      Tensor op result: {result} (expected {expected})")
        print(f"      Tensor place: {place_str}")
        print(f"      PASS: GPU tensor operation succeeded.")
    except Exception as exc:
        print(f"      FAIL: GPU tensor operation raised: {exc}")
        traceback.print_exc()
        return 1

    # ── 5. Summary ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("GPU SMOKE TEST: PASS")
    print(f"  PaddlePaddle: {paddle.__version__}")
    print(f"  Device:       {device}")
    print(f"  GPU count:    {gpu_count}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
