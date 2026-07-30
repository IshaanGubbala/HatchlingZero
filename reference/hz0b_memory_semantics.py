"""HZ-0B Phase B3: memory semantics -- the decision layer above B2's
mechanical operations.

B2's `_choose_write_slot` used one hard-coded similarity threshold (0.95)
to decide both "is this the same memory" and "does this collide with
something." The plan's B3 section says explicitly: "Do not rely on one
similarity threshold for all behaviors." This module replaces that single
threshold with separate, named control signals, still fully isolated from
any language model (matching B0-B2's isolation), operating on top of
reference.hz0b_memory_simulator's MemoryState and cosine-similarity
primitive.

Still no LM here -- these signals are pure functions of (state, key,
value, similarity), computed and testable standalone; wiring them to a
learned controller is B7's job, not this one's.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from reference.hz0b_memory_simulator import MemoryState, _cosine_similarity

# Separate, named thresholds -- replacing the single 0.95 cutoff from B2.
# Each one answers a DIFFERENT question, deliberately not shared:
UPDATE_MATCH_THRESHOLD = 0.9   # "is this close enough to be the SAME memory" (write-time)
INTERFERENCE_THRESHOLD = 0.6   # "is this close enough to plausibly cause read-time confusion with something else" (lower bar than update -- interference can happen well before two keys are considered identical)
STALENESS_AGE_HALF_LIFE = 20   # steps after which an unprotected, unreinforced memory's validity confidence has decayed by half


@dataclass(frozen=True)
class OperationDecision:
    """The B3 control signals, computed for a candidate (key, value) write
    against the current state. All in [0, 1] except erase_strength/
    slot_selection_distribution which are explicitly not squashed to a
    single scalar (kept as a distribution -- see write_probability's
    docstring)."""
    write_probability: mx.array          # [batch] -- this is plausibly a NEW fact needing a fresh slot
    update_probability: mx.array         # [batch] -- this matches an existing memory closely enough to be an update to it
    reinforce_probability: mx.array      # [batch] -- matches an existing memory AND the value is essentially unchanged (confirmation, not correction)
    contradiction_probability: mx.array  # [batch] -- matches an existing memory BUT the value differs meaningfully (a correction, distinct from reinforcement)
    erase_strength: mx.array             # [batch] -- how strongly the OLD content at the matched slot should be erased before the new value is blended in (separate from protection_strength, which governs whether the write is allowed at all)
    slot_selection_distribution: mx.array  # [batch, num_slots] -- soft distribution over which slot this operation targets, NOT collapsed to argmax here (B1 decision 5: soft addressing first)
    memory_validity_confidence: mx.array  # [batch, num_slots] -- READ-time trust estimate for every existing slot, distinct from B2's raw occupancy `confidence` (which only tracks write strength, not staleness)


def decide_operation(state: MemoryState, key: mx.array, value: mx.array) -> OperationDecision:
    """Compute the B3 control signals for a candidate write of (key, value)
    against `state`. Pure function, no mutation, no LM."""
    batch, num_slots, _ = state.keys.shape
    similarity = _cosine_similarity(key, state.keys)  # [batch, num_slots]
    occupied = (state.confidence > 1e-6).astype(mx.float32)
    similarity_occupied = similarity * occupied + (1 - occupied) * mx.array(-2.0)  # ignore empty slots for matching

    best_similarity = mx.max(similarity_occupied, axis=-1)
    best_slot = mx.argmax(similarity_occupied, axis=-1)

    # update_probability: smooth sigmoid around UPDATE_MATCH_THRESHOLD, not
    # a hard cutoff -- "close enough to be an update" is graded, not binary.
    sharpness = 15.0
    update_probability = mx.sigmoid((best_similarity - UPDATE_MATCH_THRESHOLD) * sharpness)
    write_probability = 1.0 - update_probability

    matched_value = mx.take_along_axis(state.values, best_slot[:, None, None], axis=1)[:, 0, :]
    value_change = mx.sqrt(mx.sum((value - matched_value) ** 2, axis=-1) + 1e-8)
    value_unchanged = mx.exp(-value_change)  # 1.0 when identical, decays smoothly as values diverge

    # reinforcement and contradiction are BOTH conditional on update_probability
    # being high, but split by whether the value actually changed -- two
    # observably different events sharing the same "this matches" gate,
    # exactly the B3 requirement ("reinforcement" vs "contradictory update"
    # must be separable, not the same signal).
    reinforce_probability = update_probability * value_unchanged
    contradiction_probability = update_probability * (1.0 - value_unchanged)

    # erase_strength: how much of the OLD value should be washed out before
    # blending in the new one. Full erase on a genuine contradiction; near-
    # zero erase on reinforcement (nothing to wash out, value is the same
    # anyway); moderate default for a fresh write into an empty/evicted slot.
    erase_strength = contradiction_probability + write_probability * (1.0 - occupied[mx.arange(batch), best_slot])

    # slot_selection_distribution: soft target over slots -- occupied
    # slots weighted by their match similarity (for the update path),
    # empty slots weighted for the write path, protected slots suppressed
    # entirely (mirrors B1 decision 8, but as a soft distribution here
    # rather than the hard reject/redirect B2 uses for the actual write).
    is_protected = state.protection >= 0.5
    empty_bias = (1.0 - occupied) * write_probability[:, None]
    match_bias = mx.where(similarity_occupied > INTERFERENCE_THRESHOLD, similarity_occupied, mx.array(0.0)) * update_probability[:, None]
    raw_scores = empty_bias + match_bias
    raw_scores = mx.where(is_protected, mx.array(-1e9), raw_scores)
    slot_selection_distribution = mx.softmax(raw_scores, axis=-1)

    # memory_validity_confidence: read-time trust, distinct from B2's raw
    # `confidence` (which only records write strength/occupancy, not
    # staleness). Decays with age on an explicit half-life, protection-aware
    # (protected slots stay trusted regardless of age).
    decay = mx.array(0.5) ** (state.age.astype(mx.float32) / STALENESS_AGE_HALF_LIFE)
    memory_validity_confidence = state.confidence * (decay * (1.0 - state.protection) + state.protection)

    return OperationDecision(
        write_probability=write_probability,
        update_probability=update_probability,
        reinforce_probability=reinforce_probability,
        contradiction_probability=contradiction_probability,
        erase_strength=erase_strength,
        slot_selection_distribution=slot_selection_distribution,
        memory_validity_confidence=memory_validity_confidence,
    )


# ---- Penalties (pure functions, computable standalone; wiring into an
# actual training loss is B7+'s job once there's something learned to
# penalize) ----

def excessive_write_penalty(decision: OperationDecision) -> mx.array:
    """Encourages sparse writes -- mean write probability across the batch,
    penalized directly (a write-sparsity loss, matching the plan's
    "write sparsity penalty")."""
    return mx.mean(decision.write_probability)


def uncontrolled_growth_penalty(state: MemoryState) -> mx.array:
    """Penalizes the memory saturating (average occupancy staying near 1
    across too many slots) rather than staying selectively used."""
    return mx.mean(state.confidence) ** 2


def protected_memory_corruption_penalty(before: MemoryState, after: MemoryState) -> mx.array:
    """Should be exactly 0 given B2's write() already refuses to touch
    protected slots by construction -- kept as an explicit, independent
    monitoring signal (a real assertion, not just trusting the mechanism),
    so any future change to write() that reintroduces a leak is caught by
    this penalty going non-zero rather than silently."""
    protected_mask = (before.protection >= 0.5).astype(mx.float32)
    key_drift = mx.sum((after.keys - before.keys) ** 2, axis=-1)
    value_drift = mx.sum((after.values - before.values) ** 2, axis=-1)
    return mx.sum(protected_mask * (key_drift + value_drift))


def duplicate_memory_penalty(state: MemoryState) -> mx.array:
    """Penalizes high pairwise similarity between DIFFERENT active slots'
    keys -- encourages diverse, non-redundant memory content rather than
    many slots converging on the same fact."""
    batch, num_slots, _ = state.keys.shape
    occupied = (state.confidence > 1e-6).astype(mx.float32)
    keys_norm = state.keys / mx.sqrt(mx.sum(state.keys * state.keys, axis=-1, keepdims=True) + 1e-8)
    pairwise = mx.matmul(keys_norm, keys_norm.transpose(0, 2, 1))  # [batch, num_slots, num_slots]
    occupied_pair = occupied[:, :, None] * occupied[:, None, :]
    eye = mx.eye(num_slots)[None, :, :]
    off_diagonal = occupied_pair * (1.0 - eye)
    return mx.sum(mx.maximum(pairwise, mx.array(0.0)) * off_diagonal) / mx.maximum(mx.sum(off_diagonal), mx.array(1.0))


def memory_norm_penalty(state: MemoryState, target_norm: float = 1.0) -> mx.array:
    """Penalizes key/value norms drifting from a target -- a soft version
    of legacy's hard tanh/clamp bounding (recovered_requirements.md), kept
    as a trainable penalty rather than a hard nonlinearity so gradient
    flow near the boundary isn't killed the way a hard clamp does."""
    key_norm = mx.sqrt(mx.sum(state.keys * state.keys, axis=-1) + 1e-8)
    value_norm = mx.sqrt(mx.sum(state.values * state.values, axis=-1) + 1e-8)
    occupied = (state.confidence > 1e-6).astype(mx.float32)
    key_penalty = mx.sum(occupied * (key_norm - target_norm) ** 2)
    value_penalty = mx.sum(occupied * (value_norm - target_norm) ** 2)
    return (key_penalty + value_penalty) / mx.maximum(mx.sum(occupied), mx.array(1.0))
