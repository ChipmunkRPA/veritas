"""Accelerator selection. CUDA > MPS > CPU, overridable via VERITAS_DEVICE."""
from __future__ import annotations

import torch

from .config import settings


def pick_device() -> torch.device:
    if settings.device != "auto":
        return torch.device(settings.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()


def llm_dtype() -> torch.dtype:
    """Half precision for the LLM pair (memory); classifiers stay fp32 for numerical stability."""
    if DEVICE.type in ("cuda", "mps"):
        return torch.float16
    return torch.bfloat16  # CPU: halves LLM memory; bf16 matmul is supported on x86-64 and arm64 CPUs
