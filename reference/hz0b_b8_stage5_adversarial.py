"""HZ-0B Phase B8, Stage 5 ("Adversarial memory"): the 7 scenarios the
plan names verbatim -- contradictory later information, distractors,
malicious overwrite attempts, near-identical keys, stale memories,
capacity pressure, reset boundaries -- tested directly against B2's
memory simulator (`reference/hz0b_memory_simulator.py`). No LM needed:
these are properties of the memory MECHANISM itself, testable the same
way B2/B3's own correctness suites were (fast, deterministic, no
training run), matching this phase's actual scope -- earlier B6/B7/B8
integration work already covers what happens when a real model drives
the mechanism; this is about whether the mechanism holds up under
adversarial USE patterns regardless of what's driving it.

Each scenario function returns a small, structured result dict rather
than asserting internally -- assertions live in the test suite, so the
scenario functions stay reusable (e.g. for a future real-model version
of the same adversarial tests) and the pass/fail criteria stay visible
at the call site instead of buried in this module.
"""
from __future__ import annotations

import mlx.core as mx

from reference.hz0b_memory_simulator import (
    MemoryState,
    delete,
    forget_or_decay,
    protect,
    read,
    reset,
    write,
)

NUM_SLOTS, KEY_DIM, VALUE_DIM = 8, 16, 16


def _onehot(dim: int, index: int, batch: int = 1) -> mx.array:
    row = [1.0 if i == index else 0.0 for i in range(dim)]
    return mx.array([row for _ in range(batch)])


def scenario_contradictory_later_information() -> dict:
    """Write key A -> fact1, later write key A -> fact2 (a real
    contradiction, not just an arbitrary overwrite). Correct behavior:
    reading A afterward returns fact2, not fact1 and not a blend."""
    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    key_a = _onehot(KEY_DIM, 0)
    fact1, fact2 = _onehot(VALUE_DIM, 0) * 5.0, _onehot(VALUE_DIM, 1) * 5.0
    state, slot, _ = write(state, key_a, fact1, mx.array([1.0]), step=0, slot_idx=mx.array([0]))
    state, _, _ = write(state, key_a, fact2, mx.array([1.0]), step=1, slot_idx=mx.array([0]))
    readout, _ = read(state, key_a, hard=True)
    return {"readout": readout, "fact1": fact1, "fact2": fact2}


def scenario_distractors() -> dict:
    """Real fact plus several unrelated distractor writes interspersed --
    the real fact must still be retrievable afterward, undisturbed."""
    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    real_key, real_value = _onehot(KEY_DIM, 0), _onehot(VALUE_DIM, 0) * 5.0
    state, _, _ = write(state, real_key, real_value, mx.array([1.0]), step=0, slot_idx=mx.array([0]))
    for i in range(1, 6):
        distractor_key, distractor_value = _onehot(KEY_DIM, i), _onehot(VALUE_DIM, i) * 3.0
        state, _, _ = write(state, distractor_key, distractor_value, mx.array([1.0]), step=i, slot_idx=mx.array([i]))
    readout, _ = read(state, real_key, hard=True)
    return {"readout": readout, "real_value": real_value}


def scenario_malicious_overwrite_attempt() -> dict:
    """A protected memory faces a direct, deliberate overwrite attempt
    (an oracle write targeting the SAME slot, not just a similar key) --
    must be refused, not silently succeed."""
    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    key, legit_value = _onehot(KEY_DIM, 0), _onehot(VALUE_DIM, 0) * 5.0
    state, slot, _ = write(state, key, legit_value, mx.array([1.0]), step=0, slot_idx=mx.array([0]))
    state = protect(state, slot, mx.array([1.0]))
    attacker_value = _onehot(VALUE_DIM, 1) * 99.0
    state_after_attack, _, rejected = write(state, key, attacker_value, mx.array([1.0]), step=1, slot_idx=slot)
    readout, _ = read(state_after_attack, key, hard=True)
    return {"readout": readout, "legit_value": legit_value, "attacker_value": attacker_value, "rejected": rejected}


def scenario_near_identical_keys() -> dict:
    """Two DIFFERENT facts with very similar (not identical) keys --
    correct behavior is to keep them as two distinct memories, not
    silently merge/overwrite one via the similarity-match threshold."""
    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    key_a = _onehot(KEY_DIM, 0)
    # cosine ~0.98 with key_a but a genuinely different direction, not a
    # rounding-noise duplicate -- the real "are these the same fact or
    # not" edge case, decided by B1/B2's own 0.95 match threshold.
    key_b_raw = 0.995 * key_a + (1 - 0.995**2) ** 0.5 * _onehot(KEY_DIM, 1)
    key_b = key_b_raw / mx.sqrt(mx.sum(key_b_raw * key_b_raw))
    similarity = float(mx.sum(key_a[0] * key_b[0]))
    value_a, value_b = _onehot(VALUE_DIM, 0) * 5.0, _onehot(VALUE_DIM, 1) * 5.0
    state, slot_a, _ = write(state, key_a, value_a, mx.array([1.0]), step=0, slot_idx=mx.array([0]))
    # No oracle slot_idx here -- let similarity-based routing decide,
    # since that's exactly the mechanism under test.
    state, slot_b, _ = write(state, key_b, value_b, mx.array([1.0]), step=1)
    readout_a, _ = read(state, key_a, hard=True)
    readout_b, _ = read(state, key_b, hard=True)
    return {"similarity": similarity, "slot_a": slot_a, "slot_b": slot_b, "readout_a": readout_a, "readout_b": readout_b, "value_a": value_a, "value_b": value_b}


def scenario_stale_memories(*, decay_steps: int = 20, decay_rate: float = 0.9) -> dict:
    """A memory written long ago and never refreshed -- confidence should
    genuinely decay over time. Separately (and honestly checked, not
    assumed): does staleness actually reduce this memory's influence on a
    READ, or does B2's plain similarity-based read ignore confidence
    entirely once a key matches?"""
    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    key, value = _onehot(KEY_DIM, 0), _onehot(VALUE_DIM, 0) * 5.0
    state, _, _ = write(state, key, value, mx.array([1.0]), step=0, slot_idx=mx.array([0]))
    initial_confidence = float(state.confidence[0, 0])
    for _ in range(decay_steps):
        state = forget_or_decay(state, decay_rate=decay_rate)
    decayed_confidence = float(state.confidence[0, 0])
    readout_soft, weights = read(state, key)  # soft (default) read -- does it still weight this slot fully despite low confidence?
    readout_hard, _ = read(state, key, hard=True)
    return {
        "initial_confidence": initial_confidence, "decayed_confidence": decayed_confidence,
        "readout_soft": readout_soft, "readout_hard": readout_hard, "value": value, "read_weight_on_stale_slot": float(weights[0, 0]),
    }


def scenario_capacity_pressure(*, num_facts: int = 12) -> dict:
    """Write more distinct facts than there are slots (num_facts >
    NUM_SLOTS) -- must evict sensibly (protected memories survive,
    unprotected ones may be evicted), not crash or silently corrupt."""
    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    # Protect the very first fact -- it should survive num_facts-1 more
    # competing writes even though there are only NUM_SLOTS slots total.
    protected_key, protected_value = _onehot(KEY_DIM, 0), _onehot(VALUE_DIM, 0) * 5.0
    state, protected_slot, _ = write(state, protected_key, protected_value, mx.array([1.0]), step=0, slot_idx=mx.array([0]))
    state = protect(state, protected_slot, mx.array([1.0]))
    rejected_count = 0
    for i in range(1, num_facts):
        key_i, value_i = _onehot(KEY_DIM, i % KEY_DIM), _onehot(VALUE_DIM, i % VALUE_DIM) * 2.0
        state, _, rejected = write(state, key_i, value_i, mx.array([1.0]), step=i)
        rejected_count += int(bool(rejected[0]))
    readout, _ = read(state, protected_key, hard=True)
    occupied_slots = int(mx.sum((state.confidence[0] > 1e-6).astype(mx.int32)))
    return {"readout": readout, "protected_value": protected_value, "occupied_slots": occupied_slots, "rejected_count": rejected_count, "num_facts_attempted": num_facts}


def scenario_reset_boundaries() -> dict:
    """A protected, high-confidence memory, immediately followed by
    reset() -- must be wiped completely, no leakage across the boundary,
    regardless of protection (matches B1's "full zero, matches legacy
    semantics exactly" -- reset is NOT gated by protection, unlike normal
    writes/decay)."""
    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    key, value = _onehot(KEY_DIM, 0), _onehot(VALUE_DIM, 0) * 5.0
    state, slot, _ = write(state, key, value, mx.array([1.0]), step=0, slot_idx=mx.array([0]))
    state = protect(state, slot, mx.array([1.0]))
    state_after_reset = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    readout, _ = read(state_after_reset, key)
    return {
        "readout_after_reset": readout, "value": value,
        "confidence_after_reset": float(mx.sum(state_after_reset.confidence)),
        "protection_after_reset": float(mx.sum(state_after_reset.protection)),
    }
