"""HZ-0B Phase B11: the "equal-parameter, no memory state at all" baseline
B4 named (`reference/hz0b_baselines.py`'s isolated version) but never ran
against a real frozen HZ-0A checkpoint on a real task -- this is that
real run's model half.

Exit gate under test (`plans/HZ-0B_Total_Restart_Plan.md`, Phase B11):
"HZ-0B provides a measurable advantage that cannot be explained only by
more parameters or more context." A trained read/write memory mechanism
necessarily adds trainable parameters on top of the frozen backbone; the
only way to know whether ITS SPECIFIC MECHANISM (explicit, content-
addressable, cross-position slots) matters -- as opposed to just having
extra trainable capacity anywhere in the forward pass -- is to give a
mechanism-free baseline the SAME parameter budget and see if it matches.

This adapter is a plain per-position feed-forward residual transform:
`hidden' = hidden + W2 @ relu(W1 @ hidden + b1) + b2`, applied
independently at every position (no read, no write, no cross-position
state of any kind -- each position's output depends only on that
position's own hidden state, unlike real HZ-0B memory where a write at
position t is visible to a read at position t+1). Vectorized over the
whole sequence in one matmul, unlike the memory path's per-position
Python loop, precisely because it has no sequential state to thread.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from reference.hz0b_b6_hz0a_integration import frozen_hidden_states, logits_from_hidden


@dataclass(frozen=True)
class EqualParamAdapterParams:
    w1: mx.array  # [d_model, hidden]
    b1: mx.array  # [hidden]
    w2: mx.array  # [hidden, d_model]
    b2: mx.array  # [d_model]


def param_count(d_model: int, hidden: int) -> int:
    return d_model * hidden + hidden + hidden * d_model + d_model


def init_equal_param_adapter(d_model: int, hidden: int, seed: int = 0) -> EqualParamAdapterParams:
    key = mx.random.key(seed + 3000)
    k1, k2 = mx.random.split(key)
    scale1 = (2.0 / d_model) ** 0.5
    scale2 = (2.0 / hidden) ** 0.5
    return EqualParamAdapterParams(
        w1=mx.random.normal((d_model, hidden), key=k1) * scale1,
        b1=mx.zeros((hidden,)),
        w2=mx.random.normal((hidden, d_model), key=k2) * scale2,
        b2=mx.zeros((d_model,)),
    )


def adapter_forward(params: EqualParamAdapterParams, hidden: mx.array) -> mx.array:
    """hidden: [batch, seq, d_model] -> [batch, seq, d_model]. Pure
    per-position residual MLP, no memory, no cross-position information
    flow beyond what the FROZEN backbone's own attention/recurrence
    already provided in computing `hidden` itself."""
    h = mx.maximum(hidden @ params.w1 + params.b1, 0.0)
    return hidden + (h @ params.w2 + params.b2)


def forward(model, token_ids: mx.array, *, adapter_params: EqualParamAdapterParams | None = None, states=None):
    """`adapter_params=None` -> exact no-memory, no-adapter behavior
    (the true zero-extra-parameter floor)."""
    hidden, next_states = frozen_hidden_states(model, token_ids, states)
    if adapter_params is not None:
        hidden = adapter_forward(adapter_params, hidden)
    return logits_from_hidden(model, hidden), next_states
