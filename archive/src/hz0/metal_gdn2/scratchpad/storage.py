"""
Scratchpad storage mechanisms.

Alternatives to tanh/clamp bounding:
- RMSNorm + adaptive norm clipping
- Maintains amplitude control without saturation
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple


class RMSNormStorage(nn.Module):
    """
    RMSNorm-based value storage without saturation.
    Replaces tanh clamping to prevent gradient killing.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply RMSNorm to value before storage.
        Prevents amplitude explosion while keeping gradients alive.
        """
        # RMS = sqrt(mean(x^2))
        rms = mx.sqrt(mx.mean(x ** 2, keepdims=True) + self.eps)
        normalized = x / rms
        return normalized * self.scale


def adaptive_norm_clip(
    state: mx.array,
    max_norm: float = 1.0,
    eps: float = 1e-8,
) -> mx.array:
    """
    Clip state norm adaptively without saturation.

    If ||state|| > max_norm, scale down.
    Else, keep as-is.
    """
    # Compute Frobenius norm
    norm = mx.sqrt(mx.sum(state ** 2) + eps)

    # Scale factor: min(1.0, max_norm / norm)
    scale = mx.minimum(1.0, max_norm / norm)

    return state * scale


class AdaptiveStorageBuffer:
    """
    Scratchpad storage with RMSNorm + adaptive clipping.
    Replaces tanh/clamp approach.
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        max_norm: float = 1.0,
        use_rms_norm: bool = True,
    ):
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.max_norm = max_norm
        self.use_rms_norm = use_rms_norm

        self.buffer = mx.zeros((num_slots, slot_dim))
        if use_rms_norm:
            self.rms_norm = RMSNormStorage(slot_dim)

    def store(self, slot_idx: int, value: mx.array) -> None:
        """
        Store value with RMSNorm + adaptive clipping.
        """
        # Apply RMSNorm if enabled
        if self.use_rms_norm:
            processed = self.rms_norm(value)
        else:
            processed = value

        # Adaptive norm clip
        clipped = adaptive_norm_clip(processed, max_norm=self.max_norm)

        self.buffer[slot_idx] = clipped

    def retrieve(self, slot_idx: int) -> mx.array:
        """Retrieve stored value."""
        return self.buffer[slot_idx]

    def reset(self) -> None:
        """Clear buffer."""
        self.buffer = mx.zeros_like(self.buffer)


class StorageComparison:
    """
    Compare storage mechanisms: tanh/clamp vs RMSNorm/clip.
    """

    @staticmethod
    def tanh_clamp_storage(
        value: mx.array,
        min_val: float = -1.0,
        max_val: float = 1.0,
    ) -> mx.array:
        """Original approach: tanh(-1, 1) then clamp."""
        tanhd = mx.tanh(value)
        return mx.clip(tanhd, min_val, max_val)

    @staticmethod
    def rmsnorm_clip_storage(
        value: mx.array,
        max_norm: float = 1.0,
    ) -> mx.array:
        """New approach: RMSNorm then adaptive clip."""
        rms = mx.sqrt(mx.mean(value ** 2) + 1e-6)
        normalized = value / rms
        return adaptive_norm_clip(normalized, max_norm=max_norm)
