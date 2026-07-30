"""HZ-0B Phase B6 prep: isolated read-only gated-memory integration module.

Builds ONLY the three pieces B6's plan text names ("First add only: 1.
hidden-state-to-query projection 2. memory read 3. gated memory
contribution into the residual stream") as a standalone MLX module,
tested against synthetic hidden states at HZ-0A's known-frozen width
(dim=768, per plans/HZ-0A_Progress_Tracker.md's Stage 2 config -- fixed
all session, Stage 3/4 explicitly descoped, so this width will not
change even though Stage 2 training itself is still running). No real
HZ-0A checkpoint or file is touched here -- per the plan's own gate
("STOP: any integration into HZ-0A before HZ-0A is frozen"), this stays
isolated the same way B2's simulator did, ready to wire into the real
forward pass the moment Stage 2 finishes.

Read-only by construction: no write/reinforce/update/protect/forget/
delete path exists anywhere in this module -- matches B6's own stated
scope ("Do not allow writes yet").
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from reference.hz0b_memory_simulator import MemoryState
from reference.hz0b_memory_simulator import read as memory_read


@dataclass(frozen=True)
class ReadOnlyIntegrationParams:
    query_w: mx.array            # [d_model, key_dim]
    query_b: mx.array            # [key_dim]
    gate_w: mx.array             # [d_model, d_model]
    gate_b: mx.array             # [d_model]
    value_to_hidden_w: mx.array  # [value_dim, d_model]
    value_to_hidden_b: mx.array  # [d_model]


def init_readonly_integration(d_model: int, key_dim: int, value_dim: int, seed: int = 0) -> ReadOnlyIntegrationParams:
    """Per B1 decision 14 (additive + gated residual contribution, kept
    as-is from legacy since nothing in the B0 audit implicates this
    specific equation) plus the value-to-hidden projection B1 explicitly
    flagged as deferred to B6 ("may diverge once B6 integration needs to
    match a specific residual-stream width via a value-to-hidden
    projection")."""
    key = mx.random.key(seed)
    k1, k2, k3 = mx.random.split(key, 3)
    scale_q = (2.0 / d_model) ** 0.5
    scale_g = (2.0 / d_model) ** 0.5
    scale_v = (2.0 / value_dim) ** 0.5
    return ReadOnlyIntegrationParams(
        query_w=mx.random.normal((d_model, key_dim), key=k1) * scale_q,
        query_b=mx.zeros((key_dim,)),
        gate_w=mx.random.normal((d_model, d_model), key=k2) * scale_g,
        gate_b=mx.zeros((d_model,)),
        value_to_hidden_w=mx.random.normal((value_dim, d_model), key=k3) * scale_v,
        value_to_hidden_b=mx.zeros((d_model,)),
    )


def gated_memory_read(params: ReadOnlyIntegrationParams, hidden_state: mx.array, memory_state: MemoryState) -> tuple[mx.array, mx.array]:
    """hidden_state: [batch, d_model], one sequence position.

    Read-only memory has no recurrence of its own: the memory bank is
    fixed for the whole forward pass (writes are B7's job), so applying
    this function independently to every position of a
    [batch, seq, d_model] tensor (e.g. via a batch-flattened
    [batch*seq, d_model] call) is the same function, not a different one.

    Returns (output, read_weights):
      output = hidden + sigmoid(gate_proj(hidden)) * (readout @ value_to_hidden)
    per B1 decision 14, with value_to_hidden projecting the memory's
    value_dim readout into hidden's own d_model before gating.
    """
    query = hidden_state @ params.query_w + params.query_b
    readout, read_weights = memory_read(memory_state, query)
    readout_in_hidden_space = readout @ params.value_to_hidden_w + params.value_to_hidden_b
    gate = mx.sigmoid(hidden_state @ params.gate_w + params.gate_b)
    output = hidden_state + gate * readout_in_hidden_space
    return output, read_weights
