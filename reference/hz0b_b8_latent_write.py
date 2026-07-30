"""HZ-0B Phase B8, Stage 3 ("Latent write decisions -- only final behavior
is supervised"), real integration against the frozen HZ-0A hybrid
checkpoint.

B7's real integration trained a write GATE, but the write's occurrence was
still supervised (`should_write` was a training-data label, given at a
fixed, known position) and its key/value were fixed oracle constants.
Stage 3 removes both crutches: `write_gate` is a soft, continuous,
UNSUPERVISED function of each position's own hidden state (no
`should_write` label anywhere), and `key`/`value` are themselves learned
projections of the hidden state -- the model must discover, purely from
the downstream recall loss, WHEN a position is worth writing and WHAT to
write, at every position in the sequence, not just one hand-picked one.

Writes are a SOFT, continuous blend keyed on `write_gate` itself (reusing
`hz0b_write_integration._blend_state_by_row`, the same mechanism B7 used
for its hard supervised label, with a continuous mask instead) -- a
near-zero gate leaves memory almost untouched at that position; nothing
is a hard, non-differentiable branch. This is what makes it possible to
apply a write-SPARSITY penalty (mean write_gate across all positions,
the same quantity B3's `excessive_write_penalty` computes from its own
`OperationDecision.write_probability` -- reused here in form, not by
forcing B3's full decision struct through this simpler direct-gate path)
and have gradient descent actually respond to it.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from reference.hz0b_b6_hz0a_integration import frozen_hidden_states, logits_from_hidden
from reference.hz0b_memory_simulator import MemoryState, write as memory_write
from reference.hz0b_readonly_integration import gated_memory_read
from reference.hz0b_write_integration import WriteControllerParams, _blend_state_by_row, init_write_controller


@dataclass(frozen=True)
class LatentWriteControllerParams:
    write_controller: WriteControllerParams  # reuses its read_params + write_gate_w/b; update/protect/delete gates unused in this Stage 3 v1
    key_proj_w: mx.array   # [d_model, key_dim]
    key_proj_b: mx.array   # [key_dim]
    value_proj_w: mx.array  # [d_model, value_dim]
    value_proj_b: mx.array  # [value_dim]


def init_latent_write_controller(d_model: int, key_dim: int, value_dim: int, seed: int = 0) -> LatentWriteControllerParams:
    write_controller = init_write_controller(d_model, key_dim, value_dim, seed=seed)
    key = mx.random.key(seed + 2000)
    k1, k2 = mx.random.split(key)
    scale = (2.0 / d_model) ** 0.5
    return LatentWriteControllerParams(
        write_controller=write_controller,
        key_proj_w=mx.random.normal((d_model, key_dim), key=k1) * scale,
        key_proj_b=mx.zeros((key_dim,)),
        value_proj_w=mx.random.normal((d_model, value_dim), key=k2) * scale,
        value_proj_b=mx.zeros((value_dim,)),
    )


def latent_write_and_read_step(params: LatentWriteControllerParams, hidden_state: mx.array, memory_state: MemoryState, *, step: int) -> tuple[mx.array, MemoryState, mx.array]:
    """Read happens against the PRE-write state (same write-visibility
    convention as B7 -- a write at position t is visible from t+1
    onward, matching B1 decision 7). Returns (output, new_state,
    write_gate) -- write_gate is a per-batch-row [batch] value in (0, 1),
    the exact quantity a sparsity penalty should regularize."""
    wc = params.write_controller
    output, _ = gated_memory_read(wc.read_params, hidden_state, memory_state)
    write_gate = mx.sigmoid((hidden_state @ wc.write_gate_w + wc.write_gate_b)[:, 0])
    key = hidden_state @ params.key_proj_w + params.key_proj_b
    value = hidden_state @ params.value_proj_w + params.value_proj_b
    candidate_state, _, _ = memory_write(memory_state, key, value, write_gate, step=step)
    new_state = _blend_state_by_row(memory_state, candidate_state, write_gate)
    return output, new_state, write_gate


def sequential_latent_write_and_read(params: LatentWriteControllerParams, hidden: mx.array) -> tuple[mx.array, MemoryState, mx.array]:
    """hidden: [batch, seq, d_model]. Every position gets a chance to
    write, gated continuously by its own learned `write_gate` -- no
    position is hand-picked or labeled as "the" write position, unlike
    B7. Returns (output_hidden [batch, seq, d_model], final_memory_state,
    write_gates [batch, seq])."""
    batch, seq, d_model = hidden.shape
    memory_state = MemoryState(
        keys=mx.zeros((batch, 8, params.key_proj_b.shape[0])), values=mx.zeros((batch, 8, params.value_proj_b.shape[0])),
        confidence=mx.zeros((batch, 8)), age=mx.zeros((batch, 8), dtype=mx.int32),
        protection=mx.zeros((batch, 8)), write_count=mx.zeros((batch, 8), dtype=mx.int32),
        last_write_step=mx.zeros((batch, 8), dtype=mx.int32), write_source=mx.zeros((batch, 8), dtype=mx.int32),
    )
    outputs, gates = [], []
    for t in range(seq):
        output, memory_state, write_gate = latent_write_and_read_step(params, hidden[:, t, :], memory_state, step=t)
        outputs.append(output)
        gates.append(write_gate)
    return mx.stack(outputs, axis=1), memory_state, mx.stack(gates, axis=1)


def forward(model, token_ids: mx.array, *, latent_params: LatentWriteControllerParams | None = None, states=None):
    """Full forward pass. `latent_params=None` -> exact no-memory
    behavior. Otherwise every position gets the latent write+read path."""
    hidden, next_states = frozen_hidden_states(model, token_ids, states)
    write_gates = None
    if latent_params is not None:
        hidden, _, write_gates = sequential_latent_write_and_read(latent_params, hidden)
    return logits_from_hidden(model, hidden), next_states, write_gates
