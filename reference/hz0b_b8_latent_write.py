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
from reference.hz0b_memory_simulator import MemoryState, forget_or_decay, write as memory_write
from reference.hz0b_readonly_integration import gated_memory_read
from reference.hz0b_write_integration import WriteControllerParams, _blend_state_by_row, init_write_controller


@dataclass(frozen=True)
class LatentWriteControllerParams:
    write_controller: WriteControllerParams  # reuses its read_params + write_gate_w/b; update/protect/delete gates unused in this Stage 3 v1
    key_proj_w: mx.array   # [d_model, key_dim]
    key_proj_b: mx.array   # [key_dim]
    value_proj_w: mx.array  # [d_model, value_dim]
    value_proj_b: mx.array  # [value_dim]
    occupancy_gate_w: mx.array  # [1] -- scalar weight on current max memory confidence, added to the write_gate's pre-sigmoid logit


def init_latent_write_controller(d_model: int, key_dim: int, value_dim: int, seed: int = 0, write_gate_bias_init: float = 0.0) -> LatentWriteControllerParams:
    """`write_gate_bias_init`: the write gate's bias at init (default 0.0,
    matching B6/B7's convention, sigmoid(0)=0.5 -- writes are a coin flip
    at every position before any training). A negative value (e.g. -3.0,
    sigmoid(-3)=0.047) starts the gate near-closed everywhere, so the
    network must learn to *earn* a write with a genuine gradient signal
    rather than defaulting to writing indiscriminately from step 0 -- see
    `docs/restart/hz0b_b8_stage3_results.md` section 5 for why this was
    added and the honest result (writes concentrated at sequence-start
    positions, not the informative one; this lever alone did not fix it
    at matched training budget)."""
    write_controller = init_write_controller(d_model, key_dim, value_dim, seed=seed)
    write_controller = WriteControllerParams(
        read_params=write_controller.read_params,
        write_gate_w=write_controller.write_gate_w, write_gate_b=mx.full((1,), write_gate_bias_init),
        update_gate_w=write_controller.update_gate_w, update_gate_b=write_controller.update_gate_b,
        protect_gate_w=write_controller.protect_gate_w, protect_gate_b=write_controller.protect_gate_b,
        delete_gate_w=write_controller.delete_gate_w, delete_gate_b=write_controller.delete_gate_b,
    )
    key = mx.random.key(seed + 2000)
    k1, k2 = mx.random.split(key)
    scale = (2.0 / d_model) ** 0.5
    return LatentWriteControllerParams(
        write_controller=write_controller,
        key_proj_w=mx.random.normal((d_model, key_dim), key=k1) * scale,
        key_proj_b=mx.zeros((key_dim,)),
        value_proj_w=mx.random.normal((d_model, value_dim), key=k2) * scale,
        value_proj_b=mx.zeros((value_dim,)),
        occupancy_gate_w=mx.zeros((1,)),  # starts neutral (no occupancy influence) -- the network must learn whether/how much occupancy should matter
    )


def latent_write_and_read_step(params: LatentWriteControllerParams, hidden_state: mx.array, memory_state: MemoryState, *, step: int, ste: bool = False) -> tuple[mx.array, MemoryState, mx.array]:
    """Read happens against the PRE-write state (same write-visibility
    convention as B7 -- a write at position t is visible from t+1
    onward, matching B1 decision 7). Returns (output, new_state,
    write_gate) -- write_gate is a per-batch-row [batch] value in (0, 1)
    (or exactly {0, 1} when `ste=True`), the exact quantity a sparsity
    penalty should regularize.

    The write gate's logit includes an OCCUPANCY term
    (`occupancy_gate_w * max(memory confidence across slots)`) -- without
    this, write_gate is a pure function of hidden_state alone and cannot
    possibly learn "don't bother, memory already confidently holds
    something" (or the reverse), since it never sees what memory
    currently contains. `occupancy_gate_w` starts at 0 (neutral); its
    sign and magnitude are learned.

    `ste` (2026-08-01, B1 decision 5's deferred "hard/STE routing"
    experiment, finally run): when True, the write gate is discretized to
    exactly 0 or 1 in the forward pass via a straight-through estimator
    (`hard + stop_gradient(soft - hard)`... written as
    `soft + stop_gradient(hard - soft)` below so the forward VALUE is
    `hard` while the gradient flows through `soft` as if the hard step
    were the identity function). This directly removes the continuous,
    partial-write blending traced as the likely mechanism behind both B8
    Stage 3's write-position bias
    (`docs/restart/hz0b_b8_stage3_results.md` section 3) and B11's
    train-set-size reversal (`docs/restart/hz0b_b11_evaluation_results.md`,
    "Testing the slot-capacity hypothesis") -- with `ste=True`, a write
    either fully happens or doesn't; there is no partial commit for
    gradient descent to exploit as a cheap, uninformative way to reduce
    the sparsity penalty."""
    wc = params.write_controller
    output, _ = gated_memory_read(wc.read_params, hidden_state, memory_state)
    max_confidence = mx.max(memory_state.confidence, axis=-1)
    write_logit = (hidden_state @ wc.write_gate_w + wc.write_gate_b)[:, 0] + max_confidence * params.occupancy_gate_w[0]
    write_gate_soft = mx.sigmoid(write_logit)
    if ste:
        write_gate_hard = (write_gate_soft > 0.5).astype(mx.float32)
        write_gate = write_gate_soft + mx.stop_gradient(write_gate_hard - write_gate_soft)
    else:
        write_gate = write_gate_soft
    key = hidden_state @ params.key_proj_w + params.key_proj_b
    value = hidden_state @ params.value_proj_w + params.value_proj_b
    candidate_state, _, _ = memory_write(memory_state, key, value, write_gate, step=step)
    new_state = _blend_state_by_row(memory_state, candidate_state, write_gate)
    return output, new_state, write_gate


def sequential_latent_write_and_read(params: LatentWriteControllerParams, hidden: mx.array, *, num_slots: int = 8, decay_rate: float = 1.0, ste: bool = False) -> tuple[mx.array, MemoryState, mx.array]:
    """hidden: [batch, seq, d_model]. Every position gets a chance to
    write, gated continuously by its own learned `write_gate` -- no
    position is hand-picked or labeled as "the" write position, unlike
    B7. `num_slots` matters more here than it did for B6/B7's
    oracle/supervised writes: with several slots and no real capacity
    pressure, `_choose_write_slot` routes each new (unmatched) write to
    the next empty slot, so nothing forces the network to be selective --
    a high-confidence early write becomes the LEAST likely slot to be
    evicted later (`_choose_write_slot`'s eviction score favors evicting
    LOW-confidence slots), so it can stick around regardless of
    relevance.

    `decay_rate` (default 1.0 -- no decay, matching the first Stage 3
    result that showed front-loaded, un-evictable writes) calls B2's
    `forget_or_decay` once per position: a value below 1.0 makes older
    writes' confidence erode relative to fresher ones each step, so a
    later, more relevant write has a real chance to win the eviction
    competition against an early, stale one -- the "forget" operation
    B1's contract requires but which nothing in B6/B7/this module's own
    first version ever actually exercised in a training loop.

    `ste`: see `latent_write_and_read_step`'s docstring -- threaded
    through unchanged, per-position.

    Returns (output_hidden [batch, seq, d_model], final_memory_state,
    write_gates [batch, seq])."""
    batch, seq, d_model = hidden.shape
    memory_state = MemoryState(
        keys=mx.zeros((batch, num_slots, params.key_proj_b.shape[0])), values=mx.zeros((batch, num_slots, params.value_proj_b.shape[0])),
        confidence=mx.zeros((batch, num_slots)), age=mx.zeros((batch, num_slots), dtype=mx.int32),
        protection=mx.zeros((batch, num_slots)), write_count=mx.zeros((batch, num_slots), dtype=mx.int32),
        last_write_step=mx.zeros((batch, num_slots), dtype=mx.int32), write_source=mx.zeros((batch, num_slots), dtype=mx.int32),
    )
    outputs, gates = [], []
    for t in range(seq):
        output, memory_state, write_gate = latent_write_and_read_step(params, hidden[:, t, :], memory_state, step=t, ste=ste)
        if decay_rate < 1.0:
            memory_state = forget_or_decay(memory_state, decay_rate=decay_rate)
        outputs.append(output)
        gates.append(write_gate)
    return mx.stack(outputs, axis=1), memory_state, mx.stack(gates, axis=1)


def forward(model, token_ids: mx.array | None = None, *, latent_params: LatentWriteControllerParams | None = None, states=None, num_slots: int = 8, decay_rate: float = 1.0, ste: bool = False, precomputed_hidden: mx.array | None = None):
    """Full forward pass. `latent_params=None` -> exact no-memory
    behavior. Otherwise every position gets the latent write+read path.
    `ste=False` (default) preserves every existing caller's behavior
    exactly -- opt-in, matching this project's own convention for
    non-breaking additions.

    `precomputed_hidden` (2026-08-01): skips re-running the frozen
    301M-param backbone (`frozen_hidden_states`) when the caller already
    computed it once for these exact tokens -- the backbone never
    changes across a B11-style training loop (only `latent_params`
    does), so recomputing it every gradient step was pure waste. Pass
    EITHER `token_ids` OR `precomputed_hidden`, not both. `next_states`
    is `None` in this path (unused by every current caller, which all
    discard it via `_`) since there is no fresh backbone call to derive
    it from."""
    if precomputed_hidden is not None:
        hidden, next_states = precomputed_hidden, None
    else:
        hidden, next_states = frozen_hidden_states(model, token_ids, states)
    write_gates = None
    if latent_params is not None:
        hidden, _, write_gates = sequential_latent_write_and_read(latent_params, hidden, num_slots=num_slots, decay_rate=decay_rate, ste=ste)
    return logits_from_hidden(model, hidden), next_states, write_gates
