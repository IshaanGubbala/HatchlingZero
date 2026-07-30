"""HZ-0B Phase B7: real controlled-write integration against the actual,
frozen HZ-0A hybrid checkpoint.

Where B6's real integration (`reference/hz0b_b6_hz0a_integration.py`)
populated memory via the oracle bypass BEFORE the forward pass even
started (so "read" was the only thing exercised against real hidden
states), B7 exercises the actual write path DURING a real forward pass:
a per-position sequential loop threads the memory state across the
sequence, so a write triggered by a trained, supervised write gate at one
position is what a read at a LATER position in the SAME sequence actually
retrieves. This is what B7's own text means by "the model can store and
retrieve supervised memories reliably" -- storage and retrieval both
happen inside one real forward pass, not via an external bypass.

`reference/hz0a_mlx_model.py` is still never modified. The frozen
backbone's hidden states for every position are computed in ONE pass
(GDN-2/attention state carry is a property of the frozen backbone alone,
independent of memory, since memory injection happens after all blocks
per B6's own injection point) -- only the memory bookkeeping loop below
is sequential, and it operates on tiny [batch, num_slots, dim] tensors
(reusing `reference/hz0b_write_integration.py`'s already-tested
per-position controller functions directly), not the expensive backbone
itself.
"""
from __future__ import annotations

import mlx.core as mx

from reference.hz0b_b6_hz0a_integration import frozen_hidden_states, logits_from_hidden
from reference.hz0b_memory_simulator import MemoryState
from reference.hz0b_write_integration import (
    SupervisedWriteLabel,
    WriteControllerParams,
    read_only_step,
    read_plus_supervised_write_step,
)


def sequential_write_and_read(params: WriteControllerParams, hidden: mx.array, write_labels: list[SupervisedWriteLabel | None], *, confidence_scaled: bool = False) -> tuple[mx.array, MemoryState]:
    """hidden: [batch, seq, d_model]. write_labels: one entry per sequence
    position, len == seq -- None means "no write opportunity at this
    position, read-only" (`read_only_step`), a `SupervisedWriteLabel`
    means "this position's write gate gets a chance to fire, per
    `label.should_write`" (`read_plus_supervised_write_step`).

    Threads memory_state across positions in order (write-visibility is
    token-ordered per B1 decision 7 -- a write at position t is visible
    to reads at position t+1 onward, not to the read at t itself, which
    is already how `read_plus_supervised_write_step` behaves).
    `confidence_scaled`: see `gated_memory_read`'s own docstring -- the
    fix for the bias-leakage bug this module's own write-up traced;
    default False preserves the original B7 result exactly. Returns
    (output_hidden [batch, seq, d_model], final_memory_state)."""
    batch, seq, d_model = hidden.shape
    num_slots = 8  # matches every memory_reset() call in this integration -- keep in sync if that changes
    memory_state = MemoryState(
        keys=mx.zeros((batch, num_slots, 32)), values=mx.zeros((batch, num_slots, 32)),
        confidence=mx.zeros((batch, num_slots)), age=mx.zeros((batch, num_slots), dtype=mx.int32),
        protection=mx.zeros((batch, num_slots)), write_count=mx.zeros((batch, num_slots), dtype=mx.int32),
        last_write_step=mx.zeros((batch, num_slots), dtype=mx.int32), write_source=mx.zeros((batch, num_slots), dtype=mx.int32),
    )
    outputs = []
    for t in range(seq):
        position_hidden = hidden[:, t, :]
        label = write_labels[t] if t < len(write_labels) else None
        if label is None:
            output, memory_state = read_only_step(params, position_hidden, memory_state, confidence_scaled=confidence_scaled)
        else:
            output, memory_state, _ = read_plus_supervised_write_step(params, position_hidden, memory_state, label, step=t, confidence_scaled=confidence_scaled)
        outputs.append(output)
    return mx.stack(outputs, axis=1), memory_state


def forward(model, token_ids: mx.array, *, controller_params: WriteControllerParams | None = None, write_labels: list | None = None, states=None, confidence_scaled: bool = False):
    """Full forward pass. `controller_params`/`write_labels` both None ->
    exact no-memory behavior. Both provided -> the real store-then-
    retrieve path."""
    hidden, next_states = frozen_hidden_states(model, token_ids, states)
    if controller_params is not None and write_labels is not None:
        hidden, _ = sequential_write_and_read(controller_params, hidden, write_labels, confidence_scaled=confidence_scaled)
    return logits_from_hidden(model, hidden), next_states
