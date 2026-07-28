from __future__ import annotations

from contextlib import nullcontext

import torch


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type not in {"cuda", "cpu"}:
        return nullcontext()
    if dtype not in {torch.float16, torch.bfloat16}:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def maybe_sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        return
    if device.type == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def current_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.memory_allocated(device))
    if device.type == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        current_allocated = getattr(torch.mps, "current_allocated_memory", None)
        if callable(current_allocated):
            return int(current_allocated())
    return 0
