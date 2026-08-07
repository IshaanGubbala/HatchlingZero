"""HZ-0F: MoE expert SwiGLU via MLX's native `mx.gather_mm`.

Follow-up to the E9 PMetal investigation, per a survey of 2026 MLX
developments: before spending further effort on hand-written Metal
kernels, benchmark MLX's own native grouped/gathered matmul primitive
(`mx.gather_mm`, public since MLX 0.31.2) against the existing
candidates (dense, MLX reference `moe_ffn_forward`, the hand-written
`mx.fast.metal_kernel` two-stage kernel from
`reference/hz0e_e9_mlx_native_kernel.py`).

Two variants, both real, both verified against the same toy fixtures
and real-checkpoint parity check every other kernel iteration in this
project used:

- `gather_mm_moe_forward(..., fused_gate_up=False)`: 3 `gather_mm` calls
  (gate, up, down), matching the reference's own separate gate/up
  projections.
- `gather_mm_moe_forward(..., fused_gate_up=True)`: gate and up weights
  concatenated into one `[experts, 2*expert_d_ff, dim]` tensor, ONE
  `gather_mm` call produces both projections, split afterward -- halves
  the number of gathered-weight traversals for the up-projection stage,
  per the "fused gate+up SwiGLU" optimization named in the same survey.

Routing (top-1 selection, capacity, overflow, gate weight) reuses the
exact same MLX ops as `moe_ffn_forward` -- not a reimplementation. The
overflow fallback uses the same dense-SwiGLU computation every other
kernel variant in this project uses.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from reference.hz0e_moe_contract import MoeConfig, MoeLayerParams


def _dense_swiglu(x: mx.array, gate_w: mx.array, gate_b: mx.array, up_w: mx.array, up_b: mx.array, down_w: mx.array, down_b: mx.array) -> mx.array:
    return (nn.silu(x @ gate_w.T + gate_b) * (x @ up_w.T + up_b)) @ down_w.T + down_b


def gather_mm_moe_forward(x: mx.array, params: MoeLayerParams, config: MoeConfig, *, fused_gate_up: bool = True) -> mx.array:
    """Real, complete MoE forward using `mx.gather_mm` for the expert
    SwiGLU compute. Matches `moe_ffn_forward`'s `[batch, seq, dim]` ->
    `[batch, seq, dim]` contract -- a drop-in replacement."""
    batch, seq, dim = x.shape
    n = batch * seq
    x_flat = x.reshape(n, dim)

    router_logits = x_flat @ params.router_w.T + params.router_b
    router_probs = mx.softmax(router_logits, axis=-1)
    expert_idx = mx.argmax(router_logits, axis=-1)
    gate_weight = mx.take_along_axis(router_probs, expert_idx[:, None], axis=-1)[:, 0]

    one_hot = (expert_idx[:, None] == mx.arange(config.num_experts)[None, :]).astype(mx.int32)
    running_count = mx.cumsum(one_hot, axis=0)
    rank_in_expert = mx.sum(running_count * one_hot, axis=-1) - 1

    capacity = int(mx.ceil(mx.array(config.capacity_factor * n / config.num_experts)).item())
    capacity = min(capacity, n)
    overflow = rank_in_expert >= capacity

    expert_idx_u32 = expert_idx.astype(mx.uint32)
    a = x_flat[:, None, :]  # [n, 1, dim]

    if fused_gate_up:
        w_gu = mx.concatenate([params.expert_gate_w, params.expert_up_w], axis=1)  # [E, 2*dff, dim]
        w_gu_t = w_gu.transpose(0, 2, 1)  # [E, dim, 2*dff]
        b_gu = mx.concatenate([params.expert_gate_b, params.expert_up_b], axis=1)  # [E, 2*dff]
        gu = mx.gather_mm(a, w_gu_t, rhs_indices=expert_idx_u32)[:, 0, :] + b_gu[expert_idx_u32]
        gate, up = mx.split(gu, 2, axis=-1)
    else:
        gw_t = params.expert_gate_w.transpose(0, 2, 1)
        uw_t = params.expert_up_w.transpose(0, 2, 1)
        gate = mx.gather_mm(a, gw_t, rhs_indices=expert_idx_u32)[:, 0, :] + params.expert_gate_b[expert_idx_u32]
        up = mx.gather_mm(a, uw_t, rhs_indices=expert_idx_u32)[:, 0, :] + params.expert_up_b[expert_idx_u32]

    hidden = nn.silu(gate) * up
    dw_t = params.expert_down_w.transpose(0, 2, 1)  # [E, dff, dim]
    expert_out = mx.gather_mm(hidden[:, None, :], dw_t, rhs_indices=expert_idx_u32)[:, 0, :] + params.expert_down_b[expert_idx_u32]

    gate_scale = mx.where(overflow, mx.array(1.0, dtype=mx.float32), gate_weight)
    expert_out = expert_out * gate_scale[:, None]

    fallback_out = _dense_swiglu(
        x_flat, params.fallback_gate_w, params.fallback_gate_b,
        params.fallback_up_w, params.fallback_up_b, params.fallback_down_w, params.fallback_down_b,
    )
    out = mx.where(overflow[:, None], fallback_out, expert_out)
    return out.reshape(batch, seq, dim)
