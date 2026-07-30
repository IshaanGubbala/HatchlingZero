"""HZ-0B Phase B6: real read-only memory integration against the actual,
frozen HZ-0A hybrid checkpoint.

This is the real thing the B6 prep (`reference/hz0b_readonly_integration.py`)
was built ahead of: HZ-0A's Stage 2 finished and was full-holdout evaluated
(`plans/HZ-0A_Progress_Tracker.md`, 2026-07-30), so the "STOP: any
integration into HZ-0A before HZ-0A is frozen" gate no longer applies.

`reference/hz0a_mlx_model.py` is not modified -- this module instead
reproduces its `__call__`'s embedding -> blocks -> final_norm -> tied-LM-head
path externally, with an optional read-only memory step inserted between
"all blocks done" and "final_norm", so the exact same frozen model instance
can be run both with and without memory for a real, direct comparison.
Per B6's own text ("Keep HZ-0A frozen. Do not allow writes yet."), nothing
here ever updates a `HZ0AMlxModel` parameter -- only the small
`ReadOnlyIntegrationParams` (query/gate/value-to-hidden projections) are
ever trained, and only memory *reads* happen against a memory bank that is
populated exclusively via the oracle/non-learned write bypass (B1 decision
13), never a learned write.
"""
from __future__ import annotations

import mlx.core as mx

from reference.hz0a_mlx_model import HZ0AMlxModel
from reference.hz0b_memory_simulator import MemoryState
from reference.hz0b_readonly_integration import ReadOnlyIntegrationParams
from reference.hz0b_readonly_integration import gated_memory_read as _gated_memory_read_single_position


def frozen_hidden_states(model: HZ0AMlxModel, token_ids: mx.array, states=None) -> tuple[mx.array, list]:
    """Reproduces `HZ0AMlxModel.__call__`'s embedding+blocks loop, stopping
    BEFORE `final_norm` -- this is the injection point B6's text specifies
    ("gated memory contribution into the residual stream"), i.e. after the
    backbone's own residual stream is fully formed but before the LM head."""
    x = model.embedding(token_ids)
    if states is None:
        states = [None] * len(model.blocks)
    next_states = []
    for block, state in zip(model.blocks, states):
        x, state = block(x, state)
        next_states.append(state)
    return x, next_states


def logits_from_hidden(model: HZ0AMlxModel, hidden: mx.array) -> mx.array:
    """The tail of `HZ0AMlxModel.__call__` -- final_norm + tied LM head."""
    return mx.matmul(model.final_norm(hidden), model.embedding.weight.T)


def apply_readonly_memory(params: ReadOnlyIntegrationParams, hidden: mx.array, memory_state: MemoryState, *, confidence_scaled: bool = False) -> mx.array:
    """Broadcasts the SAME (sequence-local, per B1 decision 11) memory
    bank across every position of a [batch, seq, d_model] hidden-state
    tensor, then applies B6's per-position gated read
    (`hz0b_readonly_integration.gated_memory_read`) at every position.

    Memory does not vary across the sequence in read-only mode (no writes
    happen during a forward pass here) -- what varies per-position is only
    the QUERY (derived from that position's own hidden state)."""
    batch, seq, d_model = hidden.shape
    num_slots = memory_state.keys.shape[1]
    flat_hidden = hidden.reshape(batch * seq, d_model)
    tiled_memory = MemoryState(
        keys=mx.broadcast_to(memory_state.keys[:, None], (batch, seq, num_slots, memory_state.keys.shape[-1])).reshape(batch * seq, num_slots, -1),
        values=mx.broadcast_to(memory_state.values[:, None], (batch, seq, num_slots, memory_state.values.shape[-1])).reshape(batch * seq, num_slots, -1),
        confidence=mx.broadcast_to(memory_state.confidence[:, None], (batch, seq, num_slots)).reshape(batch * seq, num_slots),
        age=mx.broadcast_to(memory_state.age[:, None], (batch, seq, num_slots)).reshape(batch * seq, num_slots),
        protection=mx.broadcast_to(memory_state.protection[:, None], (batch, seq, num_slots)).reshape(batch * seq, num_slots),
        write_count=mx.broadcast_to(memory_state.write_count[:, None], (batch, seq, num_slots)).reshape(batch * seq, num_slots),
        last_write_step=mx.broadcast_to(memory_state.last_write_step[:, None], (batch, seq, num_slots)).reshape(batch * seq, num_slots),
        write_source=mx.broadcast_to(memory_state.write_source[:, None], (batch, seq, num_slots)).reshape(batch * seq, num_slots),
    )
    flat_output, _ = _gated_memory_read_single_position(params, flat_hidden, tiled_memory, confidence_scaled=confidence_scaled)
    return flat_output.reshape(batch, seq, d_model)


def forward(model: HZ0AMlxModel, token_ids: mx.array, *, memory_params: ReadOnlyIntegrationParams | None = None, memory_state: MemoryState | None = None, states=None, confidence_scaled: bool = False) -> tuple[mx.array, list]:
    """Full forward pass. `memory_params`/`memory_state` both None -> exact
    `model(token_ids, states)` behavior (the "HZ-0A frozen, no memory" arm
    of B6's own comparison). Both provided -> the "HZ-0A frozen, read-only
    memory" arm. `confidence_scaled`: see `gated_memory_read`'s own
    docstring -- default False preserves the original B6 result exactly."""
    hidden, next_states = frozen_hidden_states(model, token_ids, states)
    if memory_params is not None and memory_state is not None:
        hidden = apply_readonly_memory(memory_params, hidden, memory_state, confidence_scaled=confidence_scaled)
    return logits_from_hidden(model, hidden), next_states
