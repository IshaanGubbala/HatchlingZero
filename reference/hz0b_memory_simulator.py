"""HZ-0B Phase B2: pure reference memory simulator.

Implements the B1 contract (docs/restart/hz0b_b1_memory_contract.md)
exactly, with no language model attached -- this is deliberately isolated
per the plan's own B2 instruction ("Before integrating with HZ-0A, build
an isolated memory simulator... The simulator must expose exact operations
and state transitions").

State is a plain, functional (immutable) structure -- every operation takes
a state in and returns a new state out, matching the rest of this
project's MLX conventions (e.g. GDN-2's forward returning next_state)
rather than in-place mutation.

Two addressing modes, both real (per contract decision 5 and 13):
- Oracle mode (`slot_idx` passed explicitly): deterministic, hard,
  non-differentiable -- used for the B2 correctness tests below, which
  need exact, verifiable state transitions, not learned routing.
- Learned mode (`slot_idx=None`): soft, differentiable addressing via
  key/query similarity -- exercised separately (see
  test_write_read_learned_mode_is_differentiable) to prove gradients
  actually flow, but not the primary vehicle for correctness testing.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx


@dataclass(frozen=True)
class MemoryState:
    keys: mx.array          # [batch, num_slots, key_dim]
    values: mx.array        # [batch, num_slots, value_dim]
    confidence: mx.array    # [batch, num_slots]
    age: mx.array           # [batch, num_slots], int32
    protection: mx.array    # [batch, num_slots]
    write_count: mx.array   # [batch, num_slots], int32
    last_write_step: mx.array  # [batch, num_slots], int32
    write_source: mx.array  # [batch, num_slots], int32 (0=supervised/oracle, 1=latent/learned)


SOURCE_SUPERVISED = 0
SOURCE_LATENT = 1

# Slots with protection at or above this are never chosen as a write
# target by similarity/eviction scoring (contract decision 8).
PROTECTION_BLOCK_THRESHOLD = 0.5


def reset(batch_size: int, num_slots: int, key_dim: int, value_dim: int) -> MemoryState:
    """Contract op: reset(). Full zero -- matches the recovered legacy
    semantics exactly (session/sequence-local, explicit, no partial state
    survives)."""
    return MemoryState(
        keys=mx.zeros((batch_size, num_slots, key_dim)),
        values=mx.zeros((batch_size, num_slots, value_dim)),
        confidence=mx.zeros((batch_size, num_slots)),
        age=mx.zeros((batch_size, num_slots), dtype=mx.int32),
        protection=mx.zeros((batch_size, num_slots)),
        write_count=mx.zeros((batch_size, num_slots), dtype=mx.int32),
        last_write_step=mx.zeros((batch_size, num_slots), dtype=mx.int32),
        write_source=mx.zeros((batch_size, num_slots), dtype=mx.int32),
    )


def _cosine_similarity(query: mx.array, keys: mx.array) -> mx.array:
    """query: [batch, dim], keys: [batch, num_slots, dim] -> [batch, num_slots].

    Epsilon added INSIDE the sqrt (not as a post-hoc max-clamp) so this
    stays differentiable at exactly-zero vectors (empty slots have
    all-zero keys) -- a max-clamp has a non-smooth kink at the threshold
    and the norm's gradient still blows up as the true norm approaches
    zero from above; adding eps under the sqrt avoids both.
    """
    query_norm = query / mx.sqrt(mx.sum(query * query, axis=-1, keepdims=True) + 1e-8)
    keys_norm = keys / mx.sqrt(mx.sum(keys * keys, axis=-1, keepdims=True) + 1e-8)
    return mx.sum(keys_norm * query_norm[:, None, :], axis=-1)


def read(state: MemoryState, query: mx.array, *, slot_idx: mx.array | None = None, hard: bool = False, confidence_weighted: bool = True) -> tuple[mx.array, mx.array]:
    """Contract op: read(query) -> (readout, read_weights).

    Content-addressable: similarity is against `state.keys` (what was
    actually written), not a fixed separate routing parameter -- the
    core B1 deviation from legacy (see hz0b_b1_memory_contract.md).

    `confidence_weighted` (default True -- a real fix, not the original
    behavior): adds `log(confidence + eps)` to each slot's similarity
    score before ranking, so a low-confidence (stale, heavily decayed)
    slot competes on worse footing than a fresh one even at equal key
    similarity, and a zero-confidence (empty) slot is pushed to
    effectively negative-infinity, excluding it outright. Fixes a real
    gap found in B8 Stage 5 (`docs/restart/hz0b_b8_stage5_results.md`
    finding 2): the original, unweighted read let a memory decayed to
    under 1% confidence still retrieve at full strength on a hard read
    -- confidence had NO effect on retrieval, only on write-eviction
    scoring. This does not change the "empty memory behaves like no
    memory" guarantee B6 depends on: an empty slot's VALUE is still
    exactly zero regardless of the weight it receives, so `readout` is
    unaffected either way when every slot is empty (the only case that
    guarantee needs)."""
    batch, num_slots, _ = state.keys.shape
    if slot_idx is not None:
        weights = mx.zeros((batch, num_slots))
        weights = mx.where(mx.arange(num_slots)[None, :] == slot_idx[:, None], mx.array(1.0), weights)
    else:
        scores = _cosine_similarity(query, state.keys)
        if confidence_weighted:
            scores = scores + mx.log(state.confidence + 1e-6)
        if hard:
            hard_idx = mx.argmax(scores, axis=-1)
            weights = (mx.arange(num_slots)[None, :] == hard_idx[:, None]).astype(mx.float32)
        else:
            weights = mx.softmax(scores, axis=-1)
    readout = mx.sum(state.values * weights[:, :, None], axis=1)
    return readout, weights


def _choose_write_slot(state: MemoryState, key: mx.array) -> tuple[mx.array, mx.array]:
    """Deterministic target-slot choice for oracle-free writes.

    Two distinct cases, per contract decision 8:
    - An existing slot's key matches (this is genuinely an update/
      overwrite attempt against a SPECIFIC memory): if that memory is
      protected, the write must be rejected outright -- redirecting
      elsewhere would silently create a second, conflicting entry for
      the same key and violate what "protected" is supposed to mean.
      If unprotected, target it directly (in-place update).
    - No existing slot matches (this is a genuinely new fact competing
      for capacity): choose the best-scoring UNPROTECTED slot (empty
      first, then lowest confidence*(1-protection), ties broken by
      oldest age); reject only if every slot is protected.

    Returns (slot_idx, forced_reject) -- forced_reject is True where the
    write must be refused regardless of which slot_idx is returned.
    """
    batch, num_slots, _ = state.keys.shape
    similarity = _cosine_similarity(key, state.keys)
    # Raised from 0.95 (a real fix, not the original value): B8 Stage 5
    # found that two GENUINELY DIFFERENT facts with cosine-similar keys
    # (0.995 -- not floating-point-identical, just correlated) were being
    # silently treated as "the same fact, update it," conflating them
    # into one slot with no protection check ever triggered
    # (docs/restart/hz0b_b8_stage5_results.md finding 1). 0.999 requires
    # near-EXACT key identity (the legitimate "same key re-presented"
    # case B1 decision 8 was actually meant for -- see
    # test_overwrite_existing_fact, which reuses the literal same key
    # object and is unaffected by this change) before treating a write as
    # an in-place update; anything less goes through the normal
    # eviction-scored "new fact competing for capacity" path instead,
    # which at least weighs confidence/protection/age rather than
    # silently overwriting.
    is_match = similarity > 0.999
    is_empty = state.confidence < 1e-6
    is_protected = state.protection >= PROTECTION_BLOCK_THRESHOLD

    has_match = mx.any(is_match, axis=-1)
    matched_slot = mx.argmax(mx.where(is_match, similarity, mx.array(-2.0)), axis=-1)
    match_is_protected = mx.take_along_axis(is_protected, matched_slot[:, None], axis=1)[:, 0]

    eviction_score = state.confidence * (1.0 - state.protection) - state.age.astype(mx.float32) * 1e-6
    eviction_score = mx.where(is_protected, mx.array(float("inf")), eviction_score)
    empty_score = mx.where(is_empty & ~is_protected, mx.array(-1.0), mx.array(float("inf")))
    eviction_slot = mx.argmin(mx.minimum(empty_score, eviction_score), axis=-1)
    all_protected = mx.all(is_protected, axis=-1)

    slot_idx = mx.where(has_match, matched_slot, eviction_slot)
    forced_reject = mx.where(has_match, match_is_protected, all_protected)
    return slot_idx, forced_reject


def write(state: MemoryState, key: mx.array, value: mx.array, strength: mx.array, *, step: int, source: int = SOURCE_LATENT, slot_idx: mx.array | None = None) -> tuple[MemoryState, mx.array, bool]:
    """Contract op: write(key, value, strength) -> (new_state, chosen_slot_idx).

    Returns a third `rejected` bool array [batch]: True where every slot
    was protected and the write had to be refused outright (contract
    decision 8), in which case new_state == state for that batch row.
    """
    batch, num_slots, _ = state.keys.shape
    if slot_idx is None:
        slot_idx, rejected = _choose_write_slot(state, key)
    else:
        target_protection = mx.take_along_axis(state.protection, slot_idx[:, None], axis=1)[:, 0]
        rejected = target_protection >= PROTECTION_BLOCK_THRESHOLD

    one_hot = (mx.arange(num_slots)[None, :] == slot_idx[:, None]).astype(mx.float32)
    write_mask = one_hot * (1.0 - rejected.astype(mx.float32))[:, None]

    new_keys = state.keys * (1 - write_mask[:, :, None]) + write_mask[:, :, None] * key[:, None, :]
    new_values = state.values * (1 - write_mask[:, :, None]) + write_mask[:, :, None] * value[:, None, :]
    new_confidence = state.confidence * (1 - write_mask) + write_mask * strength[:, None]
    new_age = mx.where(write_mask > 0, mx.zeros_like(state.age), state.age)
    new_write_count = state.write_count + write_mask.astype(mx.int32)
    new_last_write_step = mx.where(write_mask > 0, mx.full(state.last_write_step.shape, step, dtype=mx.int32), state.last_write_step)
    new_write_source = mx.where(write_mask > 0, mx.full(state.write_source.shape, source, dtype=mx.int32), state.write_source)

    new_state = replace(state, keys=new_keys, values=new_values, confidence=new_confidence, age=new_age, write_count=new_write_count, last_write_step=new_last_write_step, write_source=new_write_source)
    return new_state, slot_idx, rejected


def reinforce(state: MemoryState, slot_idx: mx.array) -> MemoryState:
    """Contract op: reinforce(slot). Confidence up, age reset, VALUE UNCHANGED."""
    batch, num_slots, _ = state.keys.shape
    mask = (mx.arange(num_slots)[None, :] == slot_idx[:, None]).astype(mx.float32)
    new_confidence = mx.minimum(state.confidence + mask * 0.2, mx.array(1.0))
    new_age = mx.where(mask > 0, mx.zeros_like(state.age), state.age)
    return replace(state, confidence=new_confidence, age=new_age)


def update(state: MemoryState, slot_idx: mx.array, new_value: mx.array) -> MemoryState:
    """Contract op: update(slot, new_value). Value replaced, key and
    PROTECTION_STRENGTH UNCHANGED -- distinct from write/reinforce."""
    batch, num_slots, _ = state.keys.shape
    mask = (mx.arange(num_slots)[None, :] == slot_idx[:, None]).astype(mx.float32)
    new_values = state.values * (1 - mask[:, :, None]) + mask[:, :, None] * new_value[:, None, :]
    return replace(state, values=new_values)


def protect(state: MemoryState, slot_idx: mx.array, strength: mx.array) -> MemoryState:
    """Contract op: protect(slot, strength). Sets protection_strength explicitly."""
    batch, num_slots, _ = state.keys.shape
    mask = (mx.arange(num_slots)[None, :] == slot_idx[:, None]).astype(mx.float32)
    new_protection = state.protection * (1 - mask) + mask * strength[:, None]
    return replace(state, protection=new_protection)


def forget_or_decay(state: MemoryState, *, decay_rate: float = 0.9) -> MemoryState:
    """Contract op: forget_or_decay(step). Global per-step decay,
    protection-aware: protected slots decay much slower (contract
    decisions 8/9 tie protection and forgetting together coherently)."""
    effective_decay = decay_rate * (1.0 - state.protection) + state.protection
    new_confidence = state.confidence * effective_decay
    new_age = state.age + 1
    return replace(state, confidence=new_confidence, age=new_age)


def delete(state: MemoryState, slot_idx: mx.array) -> MemoryState:
    """Contract op: delete(slot). Explicit hard clear -- distinct from
    natural decay, immediate and total for that slot."""
    batch, num_slots, key_dim = state.keys.shape
    mask = (mx.arange(num_slots)[None, :] == slot_idx[:, None]).astype(mx.float32)
    keep = 1 - mask
    return replace(
        state,
        keys=state.keys * keep[:, :, None],
        values=state.values * keep[:, :, None],
        confidence=state.confidence * keep,
        age=mx.where(mask > 0, mx.zeros_like(state.age), state.age),
        protection=state.protection * keep,
        write_count=mx.where(mask > 0, mx.zeros_like(state.write_count), state.write_count),
        write_source=mx.where(mask > 0, mx.zeros_like(state.write_source), state.write_source),
    )


def serialize(state: MemoryState) -> dict:
    """Contract op: serialize(). Plain arrays, no framework-specific
    objects -- suitable for the same state.json + per-array-file
    checkpoint mechanism already proven for HZ-0A this session."""
    import numpy as np
    return {field: np.asarray(getattr(state, field)) for field in ("keys", "values", "confidence", "age", "protection", "write_count", "last_write_step", "write_source")}


def restore(blob: dict) -> MemoryState:
    """Contract op: restore(). Exact inverse of serialize()."""
    return MemoryState(
        keys=mx.array(blob["keys"]),
        values=mx.array(blob["values"]),
        confidence=mx.array(blob["confidence"]),
        age=mx.array(blob["age"]),
        protection=mx.array(blob["protection"]),
        write_count=mx.array(blob["write_count"]),
        last_write_step=mx.array(blob["last_write_step"]),
        write_source=mx.array(blob["write_source"]),
    )
