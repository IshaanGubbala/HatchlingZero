"""HZ-0E E9: MoE expert SwiGLU as a native MLX custom Metal kernel.

The PMetal ctypes bridge (`reference/hz0e_e9_pmetal_integration.py`,
`restart/hz0a_pmetal/crates/hz0e-pmetal-moe-bridge`) closed E9's ~40x
end-to-end slowdown to ~12-13% by fixing a real O(dim) redundant-recompute
bug in the Metal shader, then diagnosed the REMAINING gap precisely: an
isolated single-layer bridge call costs ~1.09ms even with weights fully
GPU-resident, and three such calls (one per MoE layer) account for
essentially the entire remaining ~2.4ms end-to-end gap against the MLX
reference. That remainder is the structural cost of crossing the
Python/ctypes/numpy boundary once per MoE layer -- it forces an `mx.eval`
at each crossing and prevents MLX from fusing the whole forward pass into
one lazy graph the way the pure-MLX reference path does.

This module tests the fix for THAT: `mx.fast.metal_kernel` compiles a
custom Metal kernel that runs INSIDE MLX's own lazy execution graph --
inputs and outputs are MLX arrays the whole way through, with no numpy
conversion and no host round trip. Weight residency is automatic (MLX
arrays are already GPU-resident; passing the same `mx.array` into many
calls does not re-upload it), and the whole model's forward pass -- MoE
layers included -- can be built as ONE lazy graph and evaluated ONCE, the
same pattern `reference/hz0e_e6_integration.py::forward_e6` and the plain
dense path already use.

The expert-compute kernel itself is the SAME two-stage design as the
now-fixed Rust/Metal kernel (`restart/hz0a_pmetal/metal/moe_swiglu.metal`):
stage 1 computes each token's SwiGLU hidden activation once per (token,
dff-index); stage 2 reduces it to each (token, output-dim) scalar. Routing
(top-1 expert selection, capacity-based overflow, gate weight) reuses the
EXACT same MLX ops as `reference/hz0e_moe_contract.py::moe_ffn_forward`
(argmax / cumsum / argsort), not a reimplementation -- so this module's
only new correctness surface is the expert-compute kernel, not routing.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from reference.hz0e_moe_contract import MoeConfig, MoeLayerParams


_HIDDEN_SOURCE = """
    uint token = thread_position_in_grid.x;
    uint j = thread_position_in_grid.y;
    uint tokens_n = uint(tokens[0]);
    if (token >= tokens_n) return;
    uint dim_v = uint(dim[0]);
    uint expert_d_ff_v = uint(expert_d_ff[0]);
    uint fallback_d_ff_v = uint(fallback_d_ff[0]);
    uint capacity_v = uint(capacity[0]);
    int slot = dispatch_slot[token];
    bool fb = slot < 0;
    uint dff = fb ? fallback_d_ff_v : expert_d_ff_v;
    if (j >= dff) return;
    uint expert = fb ? 0 : uint(slot) / capacity_v;
    uint input_base = token * dim_v;
    float gate_v = fb ? fallback_biases[j] : expert_biases[expert * (2 * expert_d_ff_v + dim_v) + j];
    float up_v = fb ? fallback_biases[fallback_d_ff_v + j] : expert_biases[expert * (2 * expert_d_ff_v + dim_v) + expert_d_ff_v + j];
    for (uint i = 0; i < dim_v; ++i) {
        float xv = x[input_base + i];
        if (fb) {
            gate_v += fallback_weights[j * dim_v + i] * xv;
            up_v += fallback_weights[fallback_d_ff_v * dim_v + j * dim_v + i] * xv;
        } else {
            uint base = expert * (3 * expert_d_ff_v * dim_v);
            gate_v += expert_weights[base + j * dim_v + i] * xv;
            up_v += expert_weights[base + expert_d_ff_v * dim_v + j * dim_v + i] * xv;
        }
    }
    uint max_d_ff_v = uint(max_d_ff[0]);
    float silu_gate = gate_v / (1.0f + metal::exp(-gate_v));
    hidden[token * max_d_ff_v + j] = silu_gate * up_v;
"""

_DOWN_SOURCE = """
    uint token = thread_position_in_grid.x;
    uint out_i = thread_position_in_grid.y;
    uint tokens_n = uint(tokens[0]);
    uint dim_v = uint(dim[0]);
    if (token >= tokens_n || out_i >= dim_v) return;
    uint expert_d_ff_v = uint(expert_d_ff[0]);
    uint fallback_d_ff_v = uint(fallback_d_ff[0]);
    uint capacity_v = uint(capacity[0]);
    uint max_d_ff_v = uint(max_d_ff[0]);
    int slot = dispatch_slot[token];
    bool fb = slot < 0;
    uint dff = fb ? fallback_d_ff_v : expert_d_ff_v;
    uint expert = fb ? 0 : uint(slot) / capacity_v;
    float value = fb
        ? fallback_biases[2 * fallback_d_ff_v + out_i]
        : expert_biases[expert * (2 * expert_d_ff_v + dim_v) + 2 * expert_d_ff_v + out_i];
    uint hidden_base = token * max_d_ff_v;
    for (uint j = 0; j < dff; ++j) {
        float h = hidden[hidden_base + j];
        if (fb) value += fallback_weights[2 * fallback_d_ff_v * dim_v + out_i * fallback_d_ff_v + j] * h;
        else value += expert_weights[expert * (3 * expert_d_ff_v * dim_v) + 2 * expert_d_ff_v * dim_v + out_i * expert_d_ff_v + j] * h;
    }
    output[token * dim_v + out_i] = value;
"""

_hidden_kernel = mx.fast.metal_kernel(
    name="hz0e_moe_swiglu_hidden_mlx",
    input_names=["x", "dispatch_slot", "expert_weights", "expert_biases", "fallback_weights", "fallback_biases",
                 "capacity", "dim", "expert_d_ff", "fallback_d_ff", "tokens", "max_d_ff"],
    output_names=["hidden"],
    source=_HIDDEN_SOURCE,
)

_down_kernel = mx.fast.metal_kernel(
    name="hz0e_moe_swiglu_down_mlx",
    input_names=["dispatch_slot", "expert_weights", "expert_biases", "fallback_weights", "fallback_biases", "hidden",
                 "capacity", "dim", "expert_d_ff", "fallback_d_ff", "tokens", "max_d_ff"],
    output_names=["output"],
    source=_DOWN_SOURCE,
)


def pack_params_for_mlx_kernel(params: MoeLayerParams, config: MoeConfig) -> dict[str, mx.array]:
    """Same flat layout `reference/hz0e_e9_pmetal_integration.py::pack_params_for_bridge`
    uses, built with pure MLX ops (no numpy) so the packed weights stay
    GPU-resident MLX arrays end to end."""
    e = config.num_experts
    expert_weights = mx.concatenate([
        params.expert_gate_w.reshape(e, -1), params.expert_up_w.reshape(e, -1), params.expert_down_w.reshape(e, -1),
    ], axis=1)
    expert_biases = mx.concatenate([params.expert_gate_b, params.expert_up_b, params.expert_down_b], axis=1)
    fallback_weights = mx.concatenate([
        params.fallback_gate_w.reshape(-1), params.fallback_up_w.reshape(-1), params.fallback_down_w.reshape(-1),
    ])
    fallback_biases = mx.concatenate([params.fallback_gate_b, params.fallback_up_b, params.fallback_down_b])
    return {
        "expert_weights": expert_weights.astype(mx.float32), "expert_biases": expert_biases.astype(mx.float32),
        "fallback_weights": fallback_weights.astype(mx.float32), "fallback_biases": fallback_biases.astype(mx.float32),
    }


def mlx_native_moe_forward(x: mx.array, params: MoeLayerParams, config: MoeConfig, packed: dict[str, mx.array]) -> mx.array:
    """Real, complete MoE forward using the native MLX Metal kernel path.
    Routing (`expert_idx`/`rank_in_expert`/`capacity`/`overflow`/
    `gate_weight`) is computed with the EXACT same MLX ops as
    `moe_ffn_forward` -- this function's only new logic is building
    `dispatch_slot` from those same values and running the two-stage
    expert kernel. Everything stays MLX arrays; nothing crosses into
    numpy/ctypes. Matches `moe_ffn_forward`'s own `[batch, seq, dim]` ->
    `[batch, seq, dim]` shape contract, so it is a drop-in replacement."""
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

    dispatch_slot = mx.where(overflow, mx.array(-1, dtype=mx.int32), (expert_idx * capacity + rank_in_expert).astype(mx.int32))

    max_d_ff = max(config.expert_d_ff, config.dense_d_ff)
    scalar = lambda v: mx.array([v], dtype=mx.int32)  # noqa: E731

    hidden = _hidden_kernel(
        inputs=[x_flat, dispatch_slot, packed["expert_weights"], packed["expert_biases"],
                packed["fallback_weights"], packed["fallback_biases"],
                scalar(capacity), scalar(dim), scalar(config.expert_d_ff), scalar(config.dense_d_ff),
                scalar(n), scalar(max_d_ff)],
        grid=(n, max_d_ff, 1),
        threadgroup=(min(n, 256), 1, 1),
        output_shapes=[(n, max_d_ff)],
        output_dtypes=[mx.float32],
    )[0]

    out = _down_kernel(
        inputs=[dispatch_slot, packed["expert_weights"], packed["expert_biases"],
                packed["fallback_weights"], packed["fallback_biases"], hidden,
                scalar(capacity), scalar(dim), scalar(config.expert_d_ff), scalar(config.dense_d_ff),
                scalar(n), scalar(max_d_ff)],
        grid=(n, dim, 1),
        threadgroup=(min(n, 256), 1, 1),
        output_shapes=[(n, dim)],
        output_dtypes=[mx.float32],
    )[0]

    gate_scale = mx.where(overflow, mx.array(1.0, dtype=mx.float32), gate_weight)
    out = out * gate_scale[:, None]
    return out.reshape(batch, seq, dim)
