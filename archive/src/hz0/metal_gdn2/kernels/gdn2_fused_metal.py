"""
Fully-fused GDN-2 sequence forward via Apple Metal.

Replaces the 5 sequential MLX ops × T tokens × 24 layers of small kernel
launches with a single Metal kernel that walks the entire sequence
with thread-local recurrent state.

The backward path is intentionally kept on the chunked-VJP MLX reference:
the forward launched from training (`GDN2MetalModule.__call__` in
`gdn2_forward.py`) does NOT call this module -- it stays on the existing
MLX path. This module exposes a side-by-side forward so we can verify
numerical equivalence and benchmark. Wiring it into `gdn2_forward.py` is
gated by the runtime feature flag `USE_FUSED_METAL=1` (env var, default
off) so the running Phase 14 launcher is unaffected.

Buffer map (matches the .metal kernel arg indices):
    0..6: q, k, v, decays, erases, writes, state_in    (fp32, post-sigmoid gates)
    7..8: output, state_out                            (fp32)

NOTE on gate activations:
    decays / erases / writes are EXPECTED to be ALREADY sigmoid-applied.
    The LM caller `GDN2MetalModule.__call__` does `mx.sigmoid(...)` on the
    `to_decay_erase_write` projection before forwarding here, matching
    the convention of the chunked-MLX ref path (`gdn2_sequence_ops` also
    operates on post-sigmoid gates). We intentionally do NOT apply a
    second sigmoid inside this wrapper -- doing so would silently double-
    apply and corrupt the trainable model output.

The shape constants B, T, H, Dk, Dv are NOT passed as runtime buffers;
they are substituted directly into the Metal source string at JIT
compile time via Python `str.format`.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx

_LOG = logging.getLogger("hz0.metal_gdn2.fused")

_METAL_FILE = Path(__file__).with_name("gdn2_fused_sequence.metal")
_METAL_SOURCE_TEMPLATE = _METAL_FILE.read_text(encoding="utf-8")

_KERNEL_NAME = "gdn2_fused_fwd"

# Hard cap matches the .metal's compile-time MAX_DK / MAX_DV limits.
_HEAD_DIM_CAP = 64


@functools.lru_cache(maxsize=16)
def _build_kernel(B: int, T: int, H: int, Dk: int, Dv: int):
    """JIT-compile (and cache) a fused-metal kernel for the given shape."""
    src = _METAL_SOURCE_TEMPLATE.format(B=B, T=T, H=H, Dk=Dk, Dv=Dv)
    return mx.fast.metal_kernel(
        name=_KERNEL_NAME,
        input_names=["q", "k", "v", "d", "e", "w", "state_in"],
        output_names=["out", "state_out"],
        source=src,
        header="",
        atomic_outputs=False,
        ensure_row_contiguous=True,
    )


def _shape_value(x: int) -> mx.array:
    """Helper to convert a Python int into a 1-element uint32 MLX array.
    Kept for forward-compat; not used in the current build path because
    shape constants are substituted at JIT time."""
    return mx.array(x, dtype=mx.uint32)


# -- Public forward -----------------------------------------------------------
def gdn2_fused_forward(
    queries: mx.array,    # [B, T, H, Dk]
    keys: mx.array,       # [B, T, H, Dk]
    values: mx.array,     # [B, T, H, Dv]
    decays: mx.array,     # [B, T, H, Dk]  already-sigmoid gate (caller-applied)
    erases: mx.array,     # [B, T, H, Dk]  already-sigmoid gate (caller-applied)
    writes: mx.array,     # [B, T, H, Dv]  already-sigmoid gate (caller-applied)
    initial_state: Optional[mx.array] = None,  # [B, H, Dv, Dk]
) -> Tuple[mx.array, mx.array]:
    """One-Metal-kernel GDN-2 forward. Returns (output, final_state).

    Caps Dk, Dv at compile-time 64. For larger head dimensions, callers
    should fall back to `gdn2_sequence_ops` from `reference/gdn2_mlx.py`.

    `decays` / `erases` / `writes` MUST be sigmoid-activated by the caller.
    """
    B, T, H, Dk = queries.shape
    _, _, _, Dv = values.shape
    if max(Dk, Dv) > _HEAD_DIM_CAP:
        raise ValueError(
            f"gdn2_fused_forward caps Dk=Dv at {_HEAD_DIM_CAP}; "
            f"got Dk={Dk} Dv={Dv} (use reference/gdn2_mlx.py for larger shapes)"
        )

    if initial_state is None:
        initial_state = mx.zeros((B, H, Dv, Dk), dtype=mx.float32)

    try:
        kernel = _build_kernel(B, T, H, Dk, Dv)
    except Exception as exc:  # pragma: no cover -- defensive
        _LOG.warning(
            "fused-Metal kernel compile failed for shape "
            "(B=%d, T=%d, H=%d, Dk=%d, Dv=%d): %s; "
            "caller should fall back to MLX reference.",
            B, T, H, Dk, Dv, exc,
        )
        raise

    out, state_out = kernel(
        inputs=[queries, keys, values, decays, erases, writes, initial_state],
        output_shapes=[(B, T, H, Dv), (B, H, Dv, Dk)],
        output_dtypes=[queries.dtype, mx.float32],
        # MLX 0.32 `mx.fast.metal_kernel` interprets `grid` as TOTAL threads
        # across the grid (Apple `MTLSize`-style), NOT as a threadgroup count.
        # To get `B*H` threadgroups of `Dv` threads each → `B*H*Dv` total.
        # The kernel body reads:
        #   threadgroup_position_in_grid.x  -> (b*H + h)   range [0, B*H)
        #   thread_position_in_threadgroup.x -> lid        range [0, Dv)
        # so the existing (b, h, lid) decomposition is correct with this
        # dispatch; only the Python-side grid sizing was wrong previously,
        # causing most threads to early-return and producing near-zero outputs.
        grid=(B * H * Dv, 1, 1),
        threadgroup=(Dv, 1, 1),
    )
    return out, state_out


# Convenience: gate used by `gdn2_forward.py` to opt-in to the fused kernel.
# Default off. Phase 14 launcher is unaffected regardless of how this is set.
def fused_metal_enabled() -> bool:
    return os.environ.get("USE_FUSED_METAL", "0") == "1"


__all__ = ["gdn2_fused_forward", "fused_metal_enabled"]
