"""JBDetection — GPU environment validation.

This module enforces the GPU-only production policy:

  • PaddlePaddle is installed.
  • The installed PaddlePaddle wheel is the GPU build
    (``paddle.is_compiled_with_cuda()`` is True).
  • CUDA is actually visible to the runtime
    (``paddle.device.cuda.device_count() >= 1``).
  • A minimal tensor operation succeeds on GPU:0.
  • The resulting tensor is verified to be placed on a GPU device.

If any of these fail, :func:`validate_gpu_environment` raises
:class:`GPUEnvironmentError` with a precise diagnostic message.

The application startup sequence then aborts — there is no silent
CPU fallback in production.

For local development only, ``JBDET_ALLOW_CPU=1`` can bypass the
fatal error. This MUST NOT be enabled in production.

The function is also re-exported from :mod:`jb_detection` so callers
can run it explicitly:

    >>> from jb_detection.gpu_validation import validate_gpu_environment
    >>> validate_gpu_environment()  # raises on failure
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("jb_detection.gpu_validation")


# ── Exception ──────────────────────────────────────────────────────────

class GPUEnvironmentError(RuntimeError):
    """Raised when the GPU production policy is violated.

    The message is intended to be human-readable and actionable.
    """


# ── Result dataclass ───────────────────────────────────────────────────

@dataclass
class GPUEnvironment:
    """Snapshot of the PaddlePaddle GPU environment.

    A valid environment has ``ok=True`` and a GPU device available.
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
        """Safe dict for health endpoints."""

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


# ── Optional CPU override ──────────────────────────────────────────────

def _cpu_override_active() -> bool:
    """Return True when development CPU override is explicitly enabled."""

    return os.environ.get("JBDET_ALLOW_CPU", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# ── Public API ─────────────────────────────────────────────────────────

def validate_gpu_environment() -> GPUEnvironment:
    """Validate the production GPU-only policy.

    Returns
    -------
    GPUEnvironment
        A populated snapshot describing the environment.

    Raises
    ------
    GPUEnvironmentError
        If the GPU policy is violated and ``JBDET_ALLOW_CPU`` is not
        active.

    Notes
    -----
    Production must run with a CUDA-enabled PaddlePaddle build and
    at least one visible CUDA device.

    ``JBDET_ALLOW_CPU=1`` is provided only for local development and
    testing. It must not be enabled in production.
    """

    env = GPUEnvironment()
    errors = env.errors

    # ── 1. PaddlePaddle is importable ────────────────────────────────

    try:
        # Preload libz to work around a known PaddlePaddle crash on
        # some Python/runtime combinations where the bundled zlib
        # conflicts with the system one.
        try:
            import ctypes

            ctypes.CDLL(
                "libz.so.1",
                mode=ctypes.RTLD_GLOBAL,
            )
        except Exception:
            # Not all systems provide libz.so.1 in the same location.
            pass

        import paddle  # type: ignore

        env.paddle_version = str(paddle.__version__)

    except Exception as exc:
        errors.append(
            f"PaddlePaddle is not importable: {exc}"
        )
        return _finalize(env, errors)

    # ── 2. PaddleOCR is importable ───────────────────────────────────

    try:
        import paddleocr  # type: ignore

        env.paddleocr_version = str(
            getattr(paddleocr, "__version__", "unknown")
        )

    except Exception as exc:
        errors.append(
            f"PaddleOCR is not importable: {exc}"
        )

        # Do not immediately return.
        # Paddle GPU environment can still be diagnosed below.

    # ── 3. PaddlePaddle must be CUDA-enabled ─────────────────────────

    try:
        env.compiled_with_cuda = bool(
            paddle.is_compiled_with_cuda()
        )

    except Exception as exc:
        errors.append(
            f"paddle.is_compiled_with_cuda() failed: {exc}"
        )
        return _finalize(env, errors)

    if not env.compiled_with_cuda:
        errors.append(
            "PaddlePaddle is the CPU build "
            "(paddle.is_compiled_with_cuda()=False). "
            "Production requires a CUDA-enabled PaddlePaddle GPU build. "
            f"Detected PaddlePaddle version: {env.paddle_version}"
        )

        return _finalize(env, errors)

    # ── 4. CUDA runtime must expose at least one GPU ────────────────

    try:
        env.device = str(
            paddle.device.get_device()
        )

    except Exception as exc:
        errors.append(
            f"paddle.device.get_device() failed: {exc}"
        )
        return _finalize(env, errors)

    try:
        env.gpu_count = int(
            paddle.device.cuda.device_count()
        )

    except Exception as exc:
        errors.append(
            f"paddle.device.cuda.device_count() failed: {exc}"
        )
        env.gpu_count = 0

    if env.gpu_count < 1 or env.device.lower() == "cpu":
        errors.append(
            "PaddlePaddle is the GPU build but no CUDA device is visible "
            f"(get_device()={env.device!r}, gpu_count={env.gpu_count}). "
            "Check NVIDIA driver, CUDA runtime, NVIDIA Container Toolkit, "
            "and Docker GPU configuration."
        )

        return _finalize(env, errors)

    # ── 5. GPU name (best effort) ────────────────────────────────────

    try:
        env.gpu_name = str(
            paddle.device.cuda.get_device_name(0)
        )

    except Exception:
        env.gpu_name = ""

    # ── 6. CUDA version (best effort) ────────────────────────────────

    try:
        env.cuda_version = str(
            paddle.version.cuda()  # type: ignore[attr-defined]
        )

    except Exception:
        env.cuda_version = ""

    # ── 7. Real GPU tensor operation ─────────────────────────────────
    #
    # IMPORTANT:
    # PaddlePaddle 2.4.x does NOT provide:
    #
    #     paddle.device.guard(...)
    #
    # Therefore we explicitly select GPU:0 with:
    #
    #     paddle.device.set_device("gpu:0")
    #
    # Then we perform a real tensor operation and verify that the
    # resulting tensor is actually located on GPU.

    try:
        # Explicitly select GPU:0.
        paddle.device.set_device("gpu:0")

        # Create tensor after selecting GPU.
        x = paddle.to_tensor(
            [1.0, 2.0, 3.0]
        )

        # Real tensor operation.
        y = paddle.add(x, x)

        # Expected:
        # 2 * (1 + 2 + 3) = 12
        result = float(
            y.numpy().sum()
        )

        if abs(result - 12.0) > 1e-5:
            errors.append(
                "GPU tensor operation produced an incorrect result: "
                f"got {result}, expected 12.0"
            )

            return _finalize(env, errors)

        # Verify actual tensor placement.
        tensor_place = str(
            y.place
        )

        if "gpu" not in tensor_place.lower():
            errors.append(
                "Tensor operation completed, but the resulting tensor "
                f"is not located on GPU: place={tensor_place!r}"
            )

            return _finalize(env, errors)

        env.tensor_op_ok = True

    except Exception as exc:
        errors.append(
            f"GPU tensor operation failed: {exc}"
        )

        return _finalize(env, errors)

    # ── 8. Validation successful ─────────────────────────────────────

    env.ok = True

    return _finalize(env, errors)


# ── Internal helper ────────────────────────────────────────────────────

def _finalize(
    env: GPUEnvironment,
    errors: list,
) -> GPUEnvironment:
    """Log validation result and enforce production policy."""

    if env.ok:

        logger.info(
            "GPU validation: PASS | "
            "paddle=%s paddleocr=%s cuda=%s device=%s "
            "gpu_count=%d gpu_name=%r tensor_op=OK",
            env.paddle_version,
            env.paddleocr_version,
            env.cuda_version,
            env.device,
            env.gpu_count,
            env.gpu_name,
        )

    else:

        # Build clear multi-line diagnostic message.
        lines = [
            "GPU validation: FAILED",
            "",
            "The following problems were detected:",
        ]

        for i, err in enumerate(errors, 1):
            lines.append(
                f"  {i}. {err}"
            )

        lines.extend(
            [
                "",
                "Production GPU policy: "
                "PaddlePaddle GPU + CUDA-capable GPU required.",
                "There is NO silent CPU fallback in production.",
                "",
                "To bypass for local development ONLY "
                "(NOT for production):",
                "  export JBDET_ALLOW_CPU=1",
            ]
        )

        message = "\n".join(lines)

        logger.error(message)

        if not _cpu_override_active():

            raise GPUEnvironmentError(message)

        logger.warning(
            "JBDET_ALLOW_CPU=1 is active — proceeding in "
            "development mode with GPU validation FAILED. "
            "This MUST NOT happen in production."
        )

    return env


__all__ = [
    "GPUEnvironment",
    "GPUEnvironmentError",
    "validate_gpu_environment",
]
