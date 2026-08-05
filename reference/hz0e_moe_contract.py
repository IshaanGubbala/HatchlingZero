"""HZ-0E E1: the micro-MoE expert contract.

Per `docs/restart/hz0e_e1_contract.md` (read that doc for the reasoning
behind every choice below; this module is its real, tested
implementation, not a parallel design) and the plan's own conservative
starting point: "4 experts, top-1 routing, shared dense fallback,
selected upper MLP blocks only."

This is a REFERENCE (correctness-tier) implementation, not a
throughput-optimized one -- matching `reference/hz0c_surprise_trigger.py::masked_anchor_attention`'s
own established convention in this project ("computed via a full
O(seq^2) score matrix for correctness/testability... C8's bounded
kernel must match this reference, not the other way around"). Every
expert's output is computed for every token (dense compute), then
masked/selected per token according to real top-1 routing and real
capacity-based overflow -- correct and simple, not fast. A real
dispatch-optimized PMetal kernel is E9's job, not E1's.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class MoeConfig:
    """Static, layer-independent shape/policy parameters -- mirrors
    `reference/hz0d_fast_weights.py::FastWeightConfig`'s own
    config/state split convention."""

    dim: int = 768             # matches the frozen HZ-0A/HZ-0C/HZ-0D backbone's hidden size
    dense_d_ff: int = 2304     # the ORIGINAL block's dense FFN width (real, from the checkpoint) -- also the shared fallback's width
    num_experts: int = 4       # the plan's own conservative starting point
    expert_d_ff: int = 576     # dense_d_ff // num_experts -- see contract doc section 2 for why
    capacity_factor: float = 1.5  # see contract doc section 4
    init_seed: int = 0
    init_scale: float = 0.02


@dataclass(frozen=True)
class MoeLayerParams:
    """One MoE layer's real weights -- router, `num_experts` small
    experts, and one shared full-size dense fallback. All SwiGLU FFNs
    (gate/up/down), matching `reference/hz0a_mlx_model.py::Block`'s own
    dense-FFN shape convention exactly, so the fallback is
    architecturally IDENTICAL in shape to the block's pre-MoE FFN."""

    router_w: mx.array   # [num_experts, dim]
    router_b: mx.array   # [num_experts]
    expert_gate_w: mx.array  # [num_experts, expert_d_ff, dim]
    expert_gate_b: mx.array  # [num_experts, expert_d_ff]
    expert_up_w: mx.array    # [num_experts, expert_d_ff, dim]
    expert_up_b: mx.array    # [num_experts, expert_d_ff]
    expert_down_w: mx.array  # [num_experts, dim, expert_d_ff]
    expert_down_b: mx.array  # [num_experts, dim]
    fallback_gate_w: mx.array  # [dense_d_ff, dim]
    fallback_gate_b: mx.array  # [dense_d_ff]
    fallback_up_w: mx.array    # [dense_d_ff, dim]
    fallback_up_b: mx.array    # [dense_d_ff]
    fallback_down_w: mx.array  # [dim, dense_d_ff]
    fallback_down_b: mx.array  # [dim]


def init_moe_layer(config: MoeConfig) -> MoeLayerParams:
    """Small-random init for router and experts (nothing pretrained to
    warm-start from at the isolated-contract stage); the shared
    fallback is ALSO small-random here for isolated testing -- real E6
    integration may instead initialize the fallback by COPYING the
    real pretrained dense FFN weights it is replacing (a real, live
    option this contract does not foreclose, since the fallback's
    shape is architecturally identical to the original FFN's), but
    that is an E6 integration-time decision, not part of this static
    contract."""
    key = mx.random.key(config.init_seed)
    keys = mx.random.split(key, 11)
    scale = config.init_scale
    e, d, ff, dff = config.num_experts, config.dim, config.expert_d_ff, config.dense_d_ff
    return MoeLayerParams(
        router_w=mx.random.normal((e, d), key=keys[0]) * scale,
        router_b=mx.zeros((e,)),
        expert_gate_w=mx.random.normal((e, ff, d), key=keys[1]) * scale,
        expert_gate_b=mx.zeros((e, ff)),
        expert_up_w=mx.random.normal((e, ff, d), key=keys[2]) * scale,
        expert_up_b=mx.zeros((e, ff)),
        expert_down_w=mx.random.normal((e, d, ff), key=keys[3]) * scale,
        expert_down_b=mx.zeros((e, d)),
        fallback_gate_w=mx.random.normal((dff, d), key=keys[4]) * scale,
        fallback_gate_b=mx.zeros((dff,)),
        fallback_up_w=mx.random.normal((dff, d), key=keys[5]) * scale,
        fallback_up_b=mx.zeros((dff,)),
        fallback_down_w=mx.random.normal((d, dff), key=keys[6]) * scale,
        fallback_down_b=mx.zeros((d,)),
    )


@dataclass(frozen=True)
class MoeDiagnostics:
    """Per-call routing diagnostics -- not part of the weight contract,
    but real, structured output every caller (E2's simulator, E8's
    curriculum, E10's evaluation) needs for utilization/overflow
    reporting."""

    expert_idx: mx.array     # [N] int32 -- each token's TOP-1 chosen expert (pre-overflow)
    overflow: mx.array       # [N] bool -- True where the token exceeded its expert's capacity and used the fallback instead
    gate_weight: mx.array    # [N] float32 -- the router's softmax weight for each token's chosen expert (pre-overflow)
    expert_counts: mx.array  # [num_experts] int32 -- tokens actually served by each expert (post-overflow, i.e. excludes overflowed tokens)
    fallback_count: mx.array  # scalar int32 -- tokens served by the shared fallback (overflowed tokens)


def _swiglu(x: mx.array, gate_w: mx.array, gate_b: mx.array, up_w: mx.array, up_b: mx.array, down_w: mx.array, down_b: mx.array) -> mx.array:
    """`nn.Linear`'s own `[out_features, in_features]` weight
    convention, matching every other manual-matmul caller in this
    project -- identical SwiGLU shape to `reference/hz0a_mlx_model.py::Block.__call__`'s
    own `down(silu(gate(x)) * up(x))`."""
    return (nn.silu(x @ gate_w.T + gate_b) * (x @ up_w.T + up_b)) @ down_w.T + down_b


def moe_ffn_forward(x: mx.array, params: MoeLayerParams, config: MoeConfig) -> tuple[mx.array, MoeDiagnostics]:
    """`x`: `[batch, seq, dim]`. Real top-1 routing:

    1. `router_logits = x @ router_w.T + router_b` -> softmax -> each
       token's top-1 expert (`argmax`) and that expert's own gate
       weight (the softmax probability at the argmax index) -- the
       standard top-1 gating-weight convention.
    2. Capacity: `capacity = ceil(capacity_factor * N / num_experts)`
       tokens per expert, `N = batch * seq`. A token's RANK within its
       chosen expert's assignment queue is its position in TOKEN ORDER
       among tokens that also chose that expert (deterministic, no
       random tie-break) -- tokens ranked `>= capacity` overflow.
    3. Non-overflow tokens: `expert_output * gate_weight` (the chosen
       expert's SwiGLU output, scaled by the router's own confidence).
    4. Overflow tokens: the shared fallback's SwiGLU output, UNSCALED
       (`gate_weight` is not applied) -- the fallback's entire purpose
       is to guarantee overflow tokens are treated at least as well as
       the original (pre-MoE) dense FFN would have, and down-weighting
       it by an arbitrary top-1-of-N softmax value would undermine
       that guarantee. See contract doc section 3.

    Deterministic at inference: `argmax` (no sampling), capacity
    ranking by fixed token order (no random tie-break) -- the same
    input always produces the same routing and the same output,
    verified directly in the test suite, not assumed."""
    batch, seq, dim = x.shape
    n = batch * seq
    x_flat = x.reshape(n, dim)

    router_logits = x_flat @ params.router_w.T + params.router_b  # [N, E]
    router_probs = mx.softmax(router_logits, axis=-1)
    expert_idx = mx.argmax(router_logits, axis=-1)  # [N]
    gate_weight = mx.take_along_axis(router_probs, expert_idx[:, None], axis=-1)[:, 0]  # [N]

    one_hot = (expert_idx[:, None] == mx.arange(config.num_experts)[None, :]).astype(mx.int32)  # [N, E]
    running_count = mx.cumsum(one_hot, axis=0)  # [N, E], inclusive
    rank_in_expert = mx.sum(running_count * one_hot, axis=-1) - 1  # [N], 0-indexed rank within chosen expert's queue

    capacity = int(mx.ceil(mx.array(config.capacity_factor * n / config.num_experts)).item())
    overflow = rank_in_expert >= capacity  # [N] bool

    output = mx.zeros((n, dim))
    for e in range(config.num_experts):
        expert_out = _swiglu(
            x_flat, params.expert_gate_w[e], params.expert_gate_b[e],
            params.expert_up_w[e], params.expert_up_b[e], params.expert_down_w[e], params.expert_down_b[e],
        )
        selected = (expert_idx == e) & (~overflow)
        weight = mx.where(selected, gate_weight, mx.zeros_like(gate_weight))
        output = output + expert_out * weight[:, None]

    fallback_out = _swiglu(
        x_flat, params.fallback_gate_w, params.fallback_gate_b,
        params.fallback_up_w, params.fallback_up_b, params.fallback_down_w, params.fallback_down_b,
    )
    fallback_mask = overflow.astype(mx.float32)
    output = output + fallback_out * fallback_mask[:, None]

    expert_counts = mx.stack([mx.sum(((expert_idx == e) & (~overflow)).astype(mx.int32)) for e in range(config.num_experts)])
    diagnostics = MoeDiagnostics(
        expert_idx=expert_idx, overflow=overflow, gate_weight=gate_weight,
        expert_counts=expert_counts, fallback_count=mx.sum(overflow.astype(mx.int32)),
    )
    return output.reshape(batch, seq, dim), diagnostics


def _ffn_param_count(dim: int, d_ff: int) -> int:
    return (dim * d_ff + d_ff) * 2 + (d_ff * dim + dim)  # gate + up + down, weights and biases


def moe_layer_param_counts(config: MoeConfig) -> dict[str, int]:
    """Exact total/active parameter counts for ONE MoE layer, computed
    from real shapes -- E1's own exit gate ("exact total and active
    parameter counts are known"), not a hand-maintained estimate that
    could drift, matching `reference/hz0d_fast_weights.py::fast_state_memory_bytes`'s
    own established convention in this project."""
    dense_ffn = _ffn_param_count(config.dim, config.dense_d_ff)  # == the shared fallback's own param count
    expert_ffn = _ffn_param_count(config.dim, config.expert_d_ff)
    router = config.dim * config.num_experts + config.num_experts
    total = dense_ffn + config.num_experts * expert_ffn + router
    active_no_overflow = expert_ffn + router          # typical case: 1 expert + router
    active_worst_case = dense_ffn + router             # every token in this layer overflows to the fallback
    return {
        "dense_ffn_baseline": dense_ffn,
        "per_expert_ffn": expert_ffn,
        "all_experts_ffn": config.num_experts * expert_ffn,
        "router": router,
        "moe_layer_total": total,
        "moe_layer_active_no_overflow": active_no_overflow,
        "moe_layer_active_worst_case": active_worst_case,
    }
