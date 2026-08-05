"""HZ-0D D6: frozen-backbone integration.

Per the plan's own D6 text: "Start with fast adapters only in narrow
locations such as upper MLP blocks, memory controllers, or anchor-
attention output projections. Avoid modifying the core GDN-2 update
first. Exit gate: inactive fast weights reproduce HZ-0C behavior;
active fast weights improve adaptation."

D1's contract already named the placement: the anchor-attention OUTPUT
projection (`CausalAttention.out`) at HZ-0C's 6 `ATTENTION_INDICES`
layers, `rank=16` at `dim=768`
(`docs/restart/hz0d_d1_contract.md`). This module wires that placement
into the REAL frozen HZ-0A/HZ-0C model -- not the isolated simulator --
by reusing HZ-0C's own established pattern for a custom forward pass
that reaches into `model.blocks[i].mixer`'s real weight tensors
(`scripts/hz0c_c6_conditional_attention_eval.py::conditional_hidden`,
`reference/hz0c_surprise_trigger.py::masked_anchor_attention`), rather
than modifying `reference/hz0a_mlx_model.py` itself -- the plan's own
"avoid modifying the core... first" instruction, applied here to the
whole frozen backbone, not just the GDN-2 recurrence.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from reference.hz0d_fast_weights import FastWeightConfig, FastWeightState

ATTENTION_INDICES = (4, 9, 14, 19, 24, 29)  # matches reference/hz0c_surprise_trigger.py and hz0c_c6_conditional_attention_eval.py


def d6_fast_weight_config(*, rank: int = 16, decay_rate: float = 1.0, max_delta_norm: float = 1.0, init_seed: int = 0, init_scale: float = 0.02) -> FastWeightConfig:
    """The D1 contract's real-model config: `dim=768` (the actual HZ-0A
    hidden size, not a synthetic dimension), `rank=16`,
    `num_layers=len(ATTENTION_INDICES)=6`. Every other field keeps
    `FastWeightConfig`'s own contract defaults; exposed here as
    keyword-overridable only where D2/D3 already showed real, task-
    dependent reasons to tune them (rank/decay/clip/init)."""
    return FastWeightConfig(
        dim=768, rank=rank, num_layers=len(ATTENTION_INDICES),
        decay_rate=decay_rate, max_delta_norm=max_delta_norm,
        init_seed=init_seed, init_scale=init_scale,
    )


def fast_masked_anchor_attention(x: mx.array, trigger: mx.array, *, qkv_w: mx.array, qkv_b: mx.array, out_w: mx.array, out_b: mx.array, heads: int, fast_a: mx.array, fast_b: mx.array) -> mx.array:
    """`reference/hz0c_surprise_trigger.py::masked_anchor_attention`,
    with the output projection replaced by `out_w + fast_a @ fast_b`
    (D1's `W_effective = W_base + A_fast @ B_fast`, applied at the same
    output-projection point `apply_fast_linear` uses). Everything else
    -- causal masking, trigger-gated keys, trigger-zeroed output -- is
    byte-for-byte the same computation as the real (non-fast) version,
    so this function reduces to it EXACTLY whenever `fast_a @ fast_b`
    is exactly zero (checked directly, not assumed, in
    `tests/reference/test_hz0d_d6_integration.py`)."""
    batch, seq, dim = x.shape
    head_dim = dim // heads
    qkv = x @ qkv_w.T + qkv_b
    q, k, v = mx.split(qkv.reshape(batch, seq, 3, heads, head_dim), 3, axis=2)
    q, k, v = (mx.squeeze(t, axis=2).transpose(0, 2, 1, 3) for t in (q, k, v))
    scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(head_dim, dtype=mx.float32))
    causal_mask = mx.triu(mx.full((seq, seq), -1e9), 1)
    key_trigger_mask = (1.0 - trigger)[:, None, None, :] * -1e9
    scores = scores + causal_mask[None, None] + key_trigger_mask
    weights = mx.softmax(scores, axis=-1)
    out = mx.matmul(weights, v).transpose(0, 2, 1, 3).reshape(batch, seq, dim)
    effective_out_w = out_w + fast_a @ fast_b
    out = out @ effective_out_w.T + out_b
    return out * trigger[:, :, None]


def conditional_hidden_with_fast_weights(model, token_ids: mx.array, trigger: mx.array, fast_state: FastWeightState, config: FastWeightConfig) -> mx.array:
    """`scripts/hz0c_c6_conditional_attention_eval.py::conditional_hidden`,
    with `fast_masked_anchor_attention` in place of `masked_anchor_attention`
    at each of the 6 anchor layers -- the SAME injection point HZ-0C
    already established (backbone residual stream fully formed, stopping
    before `final_norm`), so fast weights compose with the existing
    conditional-attention graph rather than defining a new one.
    `fast_state`'s layer axis is indexed by POSITION within
    `ATTENTION_INDICES` (0..5), matching `FastWeightConfig.num_layers=6`
    -- NOT by the block's real index in `model.blocks` (4, 9, 14, ...)."""
    x = model.embedding(token_ids)
    for index, block in enumerate(model.blocks):
        if index not in ATTENTION_INDICES:
            x, _ = block(x, None)
            continue
        fast_layer = ATTENTION_INDICES.index(index)
        normed = block.norm1(x)
        anchor = fast_masked_anchor_attention(
            normed, trigger,
            qkv_w=block.mixer.qkv.weight, qkv_b=block.mixer.qkv.bias,
            out_w=block.mixer.out.weight, out_b=block.mixer.out.bias,
            heads=model.heads,
            fast_a=fast_state.a_fast[fast_layer], fast_b=fast_state.b_fast[fast_layer],
        )
        x = x + anchor
        normed2 = block.norm2(x)
        x = x + block.down(nn.silu(block.gate(normed2)) * block.up(normed2))
    return x


def logits_from_hidden(model, hidden: mx.array) -> mx.array:
    """`reference/hz0b_b6_hz0a_integration.py::logits_from_hidden` --
    final_norm + tied LM head, reused directly (not reimplemented)."""
    return mx.matmul(model.final_norm(hidden), model.embedding.weight.T)


def d6_forward_with_fast_weights(model, token_ids: mx.array, trigger: mx.array, fast_state: FastWeightState, config: FastWeightConfig) -> mx.array:
    """The full D6 forward pass: real frozen HZ-0A/HZ-0C backbone,
    real conditional anchor attention, real low-rank fast-weight delta
    at the 6 anchor layers' output projections. `model`'s own parameters
    are never written to (`update`/`.at[...].add` are never called on
    anything under `model`) -- only read, matching D1's contract
    guarantee that permanent weights never change during ordinary use."""
    return logits_from_hidden(model, conditional_hidden_with_fast_weights(model, token_ids, trigger, fast_state, config))
