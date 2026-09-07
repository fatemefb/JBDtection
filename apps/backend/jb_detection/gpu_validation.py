"""JBDetection — GPU environment validation.

This module enforces the **GPU-only** production policy:

  • PaddlePaddle is installed.
  • The installed PaddlePaddle wheel is the GPU build
    (`paddle.is_compiled_with_cuda()` is True).
  • CUDA is actually visible to the runtime
    (`paddle.device.cuda.device_count() >= 1`).
  • A minimal tensor operation succeeds on the GPU.

If any of these fail, :func:`validate_gpu_environment` raises
:class:`GPUEnvironmentError` with a precise diagnostic message. The
application startup sequence then aborts — there is **no silent CPU
fallback** in production.

The function is also re-exported from :mod:`jb_detection` so callers
can run it explicitly:

    >>> from jb_detection.gpu_validation import validate_gpu_environment
    >>> validate_gpu_environment()  # raises on failure
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("jb_detection.gpu_validation")


# ── Exception ──────────────────────────────────────────────────────────
class GPUEnvironmentError(RuntimeError):
    """Raised when the GPU production policy is violated.

    The message is intended to be human-readable and actionable — it
    should tell the operator exactly what is wrong and what to do.
    """


# ── Result dataclass (for health endpoint / logging) ───────────────────
@dataclass
class GPUEnvironment:
    """Snapshot of the PaddlePaddle GPU environment.

    All fields are populated by :func:`validate_gpu_environment`.
    A valid environment has ``ok=True`` and ``device`` starting with
    ``gpu``.
    """

    ok: bool = False
    paddle_version: str = ""
    paddleocr_version: str = ""
    compiled_with_cuda: bool = False
    device: str = ""
    gpu_count: int = 0
    gpu_name: str = ""
    cuda_version: str = ""
    tensor_op_ok: bool = False
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Safe dict for health endpoints (no sensitive info)."""
        return {
            "gpu_validation": "PASS" if self.ok else "FAIL",
            "paddle_version": self.paddle_version,
            "paddleocr_version": self.paddleocr_version,
            "compiled_with_cuda": self.compiled_with_cuda,
            "device": self.device,
            "gpu_count": self.gpu_count,
            "gpu_name": self.gpu_name,
            "cuda_version": self.cuda_version,
            "tensor_op_ok": self.tensor_op_ok,
            "errors": list(self.errors),
        }


# ── Optional override (env var) ────────────────────────────────────────
# Set JBDET_ALLOW_CPU=1 to bypass the GPU requirement (development only).
# In production this env var MUST NOT be set.
def _cpu_override_active() -> bool:
    return os.environ.get("JBDET_ALLOW_CPU", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# ── Public API ─────────────────────────────────────────────────────────
def validate_gpu_environment() -> GPUEnvironment:
    """Validate that the production GPU-only policy is satisfied.

    Returns
    -------
    GPUEnvironment
        A populated snapshot describing the environment.

    Raises
    ------
    GPUEnvironmentError
        If the GPU policy is violated AND the ``JBDET_ALLOW_CPU``
        override is NOT active. In production (no override), this is
        the expected behaviour — the application must fail loudly.
    """
    env = GPUEnvironment()
    errors = env.errors

    # ── 1. PaddlePaddle is importable ────────────────────────────────
    try:
        # Preload libz to work around a known PaddlePaddle 2.6.x crash
        # on Python 3.12+ where the bundled zlib conflicts with the
        # system one (symptom: "free(): invalid pointer" SIGABRT).
        try:
            import ctypes
            ctypes.CDLL("libz.so.1", mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass  # Not all systems have libz.so.1 (e.g. musl-based)

        import paddle  # type: ignore
        env.paddle_version = paddle.__version__
    except Exception as exc:
        errors.append(f"PaddlePaddle is not importable: {exc}")
        return _finalize(env, errors)

    # ── 2. PaddleOCR is importable ───────────────────────────────────
    try:
        import paddleocr  # type: ignore
        env.paddleocr_version = paddleocr.__version__
    except Exception as exc:
        errors.append(f"PaddleOCR is not importable: {exc}")
        # We can still validate PaddlePaddle GPU below — continue.

    # ── 3. PaddlePaddle is the CUDA-enabled build ────────────────────
    try:
        env.compiled_with_cuda = bool(paddle.is_compiled_with_cuda())
    except Exception as exc:
        errors.append(f"paddle.is_compiled_with_cuda() failed: {exc}")
        return _finalize(env, errors)

    if not env.compiled_with_cuda:
        errors.append(
            "PaddlePaddle is the CPU build (paddle.is_compiled_with_cuda()=False). "
            "Production requires paddlepaddle-gpu==2.6.2. "
            "Uninstall the CPU wheel:  pip uninstall paddlepaddle -y  "
            "then install the GPU wheel:  pip install paddlepaddle-gpu==2.6.2"
        )
        return _finalize(env, errors)

    # ── 4. CUDA runtime sees at least one GPU ────────────────────────
    try:
        env.device = str(paddle.device.get_device())
    except Exception as exc:
        errors.append(f"paddle.device.get_device() failed: {exc}")
        return _finalize(env, errors)

    try:
        env.gpu_count = int(paddle.device.cuda.device_count())
    except Exception as exc:
        errors.append(f"paddle.device.cuda.device_count() failed: {exc}")
        env.gpu_count = 0

    if env.gpu_count < 1 or env.device == "cpu":
        errors.append(
            "PaddlePaddle is the GPU build but no CUDA device is visible "
            f"(get_device()={env.device!r}, gpu_count={env.gpu_count}). "
            "Check NVIDIA driver + CUDA toolkit + (for Docker) NVIDIA Container Toolkit "
            "and `--gpus all` on docker run / `deploy.resources.reservations.devices` in compose."
        )
        return _finalize(env, errors)

    # ── 5. GPU name (best-effort) ────────────────────────────────────
    try:
        # paddle.device.cuda.get_device_name() is available in PaddlePaddle 2.x
        env.gpu_name = str(paddle.device.cuda.get_device_name(0))
    except Exception:
        env.gpu_name = ""

    # ── 6. CUDA version (best-effort, from runtime) ─────────────────
    try:
        # paddle.version.cuda() returns the CUDA runtime version the
        # wheel was compiled against.
        env.cuda_version = str(paddle.version.cuda())  # type: ignore[attr-defined]
    except Exception:
        env.cuda_version = ""

    # ── 7. A minimal tensor operation actually runs on the GPU ──────
    try:
        # Force placement on GPU:0 (do not rely on get_device() default,
        # because a misconfigured environment can silently fall back to
        # CPU even with the GPU build installed).
        with paddle.device.guard("gpu:0"):
            x = paddle.to_tensor([1.0, 2.0, 3.0])
            y = paddle.add(x, x)
            result = float(y.numpy().sum())  # 2*(1+2+3) = 12
        if abs(result - 12.0) > 1e-5:
            errors.append(
                f"GPU tensor op produced wrong result: got {result}, expected 12.0"
            )
            return _finalize(env, errors)
        env.tensor_op_ok = True
    except Exception as exc:
        errors.append(f"GPU tensor operation failed: {exc}")
        return _finalize(env, errors)

    env.ok = True
    return _finalize(env, errors)


# ── Internal helper ────────────────────────────────────────────────────
def _finalize(env: GPUEnvironment, errors: list) -> GPUEnvironment:
    """Log the validation result and raise if production policy is violated."""
    if env.ok:
        logger.info(
            "GPU validation: PASS | paddle=%s paddleocr=%s cuda=%s device=%s "
            "gpu_count=%d gpu_name=%r cuda_version=%s tensor_op=OK",
            env.paddle_version, env.paddleocr_version, env.cuda_version,
            env.device, env.gpu_count, env.gpu_name, env.cuda_version,
        )
    else:
        # Build a clear, multi-line error message
        lines = ["GPU validation: FAILED", ""]
        lines.append("The following problems were detected:")
        for i, err in enumerate(errors, 1):
            lines.append(f"  {i}. {err}")
        lines.append("")
        lines.append("Production GPU policy: PaddlePaddle GPU + CUDA-capable GPU required.")
        lines.append("There is NO silent CPU fallback in production.")
        lines.append("")
        lines.append("To bypass for local development ONLY (NOT for production):")
        lines.append("  export JBDET_ALLOW_CPU=1")
        message = "\n".join(lines)

        logger.error(message)

        if not _cpu_override_active():
            raise GPUEnvironmentError(message)
        else:
            logger.warning(
                "JBDET_ALLOW_CPU=1 is active — proceeding in development mode "
                "with GPU validation FAILED. This MUST NOT happen in production."
            )

    return env


__all__ = [
    "GPUEnvironment",
    "GPUEnvironmentError",
    "validate_gpu_environment",
]
