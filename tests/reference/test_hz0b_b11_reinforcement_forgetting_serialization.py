"""HZ-0B B11 regression test: reinforcement/forgetting/serialization
accuracy (scripts/hz0b_b11_reinforcement_forgetting_serialization.py).
Locks in the real, swept result: reinforced facts stay retrievable,
unreinforced facts decay into unretrievability by step 20 (not a
floor artifact -- step 10 shows a genuine partial transition), and
serialize/restore is bit-exact at every point tested."""
from __future__ import annotations

import mlx.core as mx

from reference.hz0b_memory_simulator import (
    forget_or_decay, read, reinforce, reset, restore, serialize, write,
)

NUM_SLOTS, KEY_DIM, VALUE_DIM = 8, 32, 32
NUM_FACTS = 6
REINFORCED_IDX = [0, 1, 2]
UNREINFORCED_IDX = [3, 4, 5]
DECAY_RATE = 0.85
REINFORCE_EVERY = 10
SEED = 555


def _build_session(steps: int):
    mx.random.seed(SEED)
    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    keys = mx.random.normal((NUM_FACTS, KEY_DIM))
    values = mx.random.normal((NUM_FACTS, VALUE_DIM))
    for i in range(NUM_FACTS):
        state, _, rejected = write(state, keys[i:i + 1], values[i:i + 1], mx.array([1.0]), step=0, slot_idx=mx.array([i]))
        assert not bool(rejected[0])
    for step in range(1, steps + 1):
        state = forget_or_decay(state, decay_rate=DECAY_RATE)
        if step % REINFORCE_EVERY == 0:
            for idx in REINFORCED_IDX:
                state = reinforce(state, slot_idx=mx.array([idx]))
    return state, keys


def _accuracy(state, keys) -> dict[int, bool]:
    out = {}
    for i in range(NUM_FACTS):
        _, weights = read(state, keys[i:i + 1], hard=True, confidence_weighted=True)
        out[i] = int(mx.argmax(weights[0])) == i
    return out


def test_reinforced_facts_stay_retrievable_at_50_steps():
    state, keys = _build_session(50)
    acc = _accuracy(state, keys)
    assert all(acc[i] for i in REINFORCED_IDX), "reinforced facts must remain retrievable"


def test_unreinforced_facts_saturate_to_unretrievable_by_step_20():
    state, keys = _build_session(20)
    acc = _accuracy(state, keys)
    assert not any(acc[i] for i in UNREINFORCED_IDX), "unreinforced facts should be unretrievable by step 20 at decay_rate=0.85"


def test_forgetting_is_gradual_not_an_instant_cliff():
    """At step 10 (before full saturation), unreinforced facts are NOT
    uniformly already gone -- confirms step-20 saturation is a real
    decay curve, not an artifact of decay_rate being trivially harsh."""
    state, keys = _build_session(10)
    acc = _accuracy(state, keys)
    assert any(acc[i] for i in UNREINFORCED_IDX), "step 10 should show partial retrievability, not already-total forgetting"


def test_serialize_restore_bit_exact_after_decay_and_reinforcement():
    state, keys = _build_session(50)
    blob = serialize(state)
    restored = restore(blob)
    for field in ("keys", "values", "confidence", "age", "protection", "write_count", "last_write_step", "write_source"):
        assert bool(mx.array_equal(getattr(state, field), getattr(restored, field))), f"{field} not bit-exact after restore"
    assert _accuracy(state, keys) == _accuracy(restored, keys)
    for i in range(NUM_FACTS):
        r_orig, _ = read(state, keys[i:i + 1], hard=True, confidence_weighted=True)
        r_restored, _ = read(restored, keys[i:i + 1], hard=True, confidence_weighted=True)
        assert bool(mx.array_equal(r_orig, r_restored)), f"fact {i} readout not bit-exact after restore"
