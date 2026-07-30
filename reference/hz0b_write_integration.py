"""HZ-0B Phase B7 prep: isolated supervised-write controller module.

B7 ("Add explicit write controls while keeping HZ-0A frozen") itself
needs real HZ-0A hidden states to train against, so real B7 stays
blocked on B5/frozen-HZ-0A like B6's real integration does. What CAN be
built in isolation now, same pattern as B6's `hz0b_readonly_integration.py`,
is the controller machinery itself: the gates B7's plan text names
("write controller, update controller, protection controller") plus
B6's query projection and read gate, composed into the three comparison
modes B7's own text specifies ("read only", "read plus supervised write",
"read plus write plus update"), tested against synthetic hidden states
and synthetic supervised labels (whether to write, key, value, whether to
protect, whether to update, whether to delete -- B7's own listed data
spec, verbatim).

Design note on gates vs. labels: each controller (write/update/protect/
delete) computes a learned, differentiable gate value from hidden_state
-- this is what real B7 will train (against the supervised label, via a
loss) once a frozen HZ-0A exists. But a random-initialized gate has not
learned anything yet, so this prep module uses the supervised LABEL
itself (hard 0/1) to control whether a state transition structurally
happens, and keeps each gate's continuous value as a returned diagnostic
only. This split matters for correctness: B2's write() always commits
key/value into the matched slot once called (a `strength` of 0 only
zeroes the resulting confidence, it does not skip the key/value write) --
gating on an untrained, near-0.5 gate value instead of the label would
let "should not write" rows silently corrupt a slot's stored content.
Using the label to gate row-wise state selection sidesteps that
entirely and keeps this module correct independent of whether any
gate has been trained.

No real HZ-0A checkpoint or file is touched -- per the plan's own STOP
gate, exactly like B6.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import mlx.core as mx

from reference.hz0b_memory_simulator import MemoryState
from reference.hz0b_memory_simulator import delete as memory_delete
from reference.hz0b_memory_simulator import protect as memory_protect
from reference.hz0b_memory_simulator import update as memory_update
from reference.hz0b_memory_simulator import write as memory_write
from reference.hz0b_readonly_integration import (
    ReadOnlyIntegrationParams,
    gated_memory_read,
    init_readonly_integration,
)


@dataclass(frozen=True)
class WriteControllerParams:
    read_params: ReadOnlyIntegrationParams
    write_gate_w: mx.array     # [d_model, 1]
    write_gate_b: mx.array     # [1]
    update_gate_w: mx.array    # [d_model, 1]
    update_gate_b: mx.array    # [1]
    protect_gate_w: mx.array   # [d_model, 1]
    protect_gate_b: mx.array   # [1]
    delete_gate_w: mx.array    # [d_model, 1]
    delete_gate_b: mx.array    # [1]


def init_write_controller(d_model: int, key_dim: int, value_dim: int, seed: int = 0) -> WriteControllerParams:
    read_params = init_readonly_integration(d_model, key_dim, value_dim, seed=seed)
    key = mx.random.key(seed + 1000)
    keys = mx.random.split(key, 4)
    scale = (2.0 / d_model) ** 0.5
    return WriteControllerParams(
        read_params=read_params,
        write_gate_w=mx.random.normal((d_model, 1), key=keys[0]) * scale,
        write_gate_b=mx.zeros((1,)),
        update_gate_w=mx.random.normal((d_model, 1), key=keys[1]) * scale,
        update_gate_b=mx.zeros((1,)),
        protect_gate_w=mx.random.normal((d_model, 1), key=keys[2]) * scale,
        protect_gate_b=mx.zeros((1,)),
        delete_gate_w=mx.random.normal((d_model, 1), key=keys[3]) * scale,
        delete_gate_b=mx.zeros((1,)),
    )


def _gate(hidden_state: mx.array, w: mx.array, b: mx.array) -> mx.array:
    return mx.sigmoid((hidden_state @ w + b)[:, 0])


@dataclass(frozen=True)
class SupervisedWriteLabel:
    """B7's own required supervised-write data spec, verbatim: whether to
    write, the key, the value, whether to protect, whether to update,
    whether to delete."""
    should_write: mx.array    # [batch], hard 0/1
    key: mx.array             # [batch, key_dim]
    value: mx.array           # [batch, value_dim]
    should_protect: mx.array  # [batch], hard 0/1
    should_update: mx.array   # [batch], hard 0/1
    should_delete: mx.array   # [batch], hard 0/1
    target_slot: mx.array | None = None  # oracle slot for update/protect/delete, None -> similarity-addressed


def _blend_state_by_row(old: MemoryState, candidate: MemoryState, keep_candidate: mx.array) -> MemoryState:
    """Row-wise select between `old` and `candidate` per batch element.
    `keep_candidate` is [batch], hard 0/1 -- 1 keeps the candidate's row
    (the operation applied), 0 reverts that row to `old` exactly (the
    operation is a correct no-op for that row, not a partial/corrupting
    one)."""
    updated = {}
    for f in fields(old):
        old_val = getattr(old, f.name)
        new_val = getattr(candidate, f.name)
        mask = keep_candidate.reshape([keep_candidate.shape[0]] + [1] * (old_val.ndim - 1))
        mask = mask.astype(old_val.dtype) if "int" in str(old_val.dtype) else mask
        updated[f.name] = old_val * (1 - mask) + new_val * mask
    return MemoryState(**updated)


def read_only_step(params: WriteControllerParams, hidden_state: mx.array, memory_state: MemoryState, *, confidence_scaled: bool = False) -> tuple[mx.array, MemoryState]:
    """Compare-mode 1: "read only" -- B6's path, memory never changes.
    `confidence_scaled`: see `gated_memory_read`'s own docstring -- a real
    fix for the bias-leakage bug traced in B7's real-integration write-up;
    default False preserves every existing caller's exact prior
    behavior."""
    output, _ = gated_memory_read(params.read_params, hidden_state, memory_state, confidence_scaled=confidence_scaled)
    return output, memory_state


def read_plus_supervised_write_step(params: WriteControllerParams, hidden_state: mx.array, memory_state: MemoryState, label: SupervisedWriteLabel, *, step: int, confidence_scaled: bool = False) -> tuple[mx.array, MemoryState, mx.array]:
    """Compare-mode 2: "read plus supervised write" -- read happens
    against the PRE-write state (this step's own write isn't visible to
    its own read; only later steps see it, matching B1 decision 7's
    token-ordered write-visibility). The write gate is computed (returned
    as a diagnostic -- this is what real B7 trains against
    `label.should_write` via a loss) but `label.should_write` itself
    controls whether the row's state actually changes, per this module's
    design note above."""
    output, _ = gated_memory_read(params.read_params, hidden_state, memory_state, confidence_scaled=confidence_scaled)
    write_gate = _gate(hidden_state, params.write_gate_w, params.write_gate_b)
    candidate_state, _, _ = memory_write(memory_state, label.key, label.value, write_gate, step=step, slot_idx=label.target_slot)
    new_state = _blend_state_by_row(memory_state, candidate_state, label.should_write)
    return output, new_state, write_gate


def read_plus_write_plus_update_step(params: WriteControllerParams, hidden_state: mx.array, memory_state: MemoryState, label: SupervisedWriteLabel, *, step: int, confidence_scaled: bool = False) -> tuple[mx.array, MemoryState, dict]:
    """Compare-mode 3: "read plus write plus update" -- adds the
    update/protect/delete controllers on top of mode 2, each gated
    independently by its own supervised label (per the B1 contract's
    insistence that reinforce/update/protect stay separate, observably
    different operations rather than one collapsed knob)."""
    output, written_state, write_gate = read_plus_supervised_write_step(params, hidden_state, memory_state, label, step=step, confidence_scaled=confidence_scaled)

    if label.target_slot is not None:
        slot_idx = label.target_slot
    else:
        _, read_weights = gated_memory_read(params.read_params, hidden_state, written_state, confidence_scaled=confidence_scaled)
        slot_idx = mx.argmax(read_weights, axis=-1)

    update_gate = _gate(hidden_state, params.update_gate_w, params.update_gate_b)
    candidate_updated = memory_update(written_state, slot_idx, label.value)
    updated_state = _blend_state_by_row(written_state, candidate_updated, label.should_update)

    protect_gate = _gate(hidden_state, params.protect_gate_w, params.protect_gate_b)
    candidate_protected = memory_protect(updated_state, slot_idx, protect_gate)
    protected_state = _blend_state_by_row(updated_state, candidate_protected, label.should_protect)

    delete_gate = _gate(hidden_state, params.delete_gate_w, params.delete_gate_b)
    candidate_deleted = memory_delete(protected_state, slot_idx)
    final_state = _blend_state_by_row(protected_state, candidate_deleted, label.should_delete)

    gates = {"write": write_gate, "update": update_gate, "protect": protect_gate, "delete": delete_gate}
    return output, final_state, gates
