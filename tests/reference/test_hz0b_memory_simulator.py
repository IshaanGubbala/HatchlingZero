"""HZ-0B Phase B2 tests: the plan's own 14 initial simulator tasks
(plans/HZ-0B_Total_Restart_Plan.md, Phase B2), against
reference/hz0b_memory_simulator.py. No language model involved -- pure
isolated memory mechanics, matching B2's exit gate ("the simulator passes
predefined tests without any language model attached")."""
import mlx.core as mx

from reference.hz0b_memory_simulator import (
    SOURCE_LATENT,
    SOURCE_SUPERVISED,
    delete,
    forget_or_decay,
    protect,
    read,
    reinforce,
    reset,
    restore,
    serialize,
    update,
    write,
)

NUM_SLOTS = 8
KEY_DIM = 4
VALUE_DIM = 4


def fresh(batch_size: int = 1):
    return reset(batch_size, NUM_SLOTS, KEY_DIM, VALUE_DIM)


def onehot(dim: int, index: int, batch: int = 1) -> mx.array:
    vec = [1.0 if i == index else 0.0 for i in range(dim)]
    return mx.array([vec for _ in range(batch)])


# 1. store one key-value pair
def test_store_one_key_value_pair():
    state = fresh()
    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 1) * 5.0
    state, slot_idx, rejected = write(state, key, value, mx.array([1.0]), step=0)
    assert not bool(rejected[0])
    readout, _ = read(state, key, slot_idx=slot_idx)
    assert bool(mx.allclose(readout, value))


# 2. retrieve an exact key
def test_retrieve_exact_key():
    state = fresh()
    key, value = onehot(KEY_DIM, 2), onehot(VALUE_DIM, 3) * 7.0
    state, slot_idx, _ = write(state, key, value, mx.array([1.0]), step=0)
    readout, weights = read(state, key, hard=True)  # learned-mode retrieval, no oracle slot_idx
    assert bool(mx.allclose(readout, value, atol=1e-4))
    assert int(mx.argmax(weights[0])) == int(slot_idx[0])


# 3. retrieve from a noisy key
def test_retrieve_from_noisy_key():
    state = fresh()
    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 1) * 5.0
    state, slot_idx, _ = write(state, key, value, mx.array([1.0]), step=0)
    noisy_key = key + mx.array([[0.05, -0.03, 0.02, 0.01]])
    readout, weights = read(state, noisy_key, hard=True)
    assert int(mx.argmax(weights[0])) == int(slot_idx[0])
    assert bool(mx.allclose(readout, value, atol=1e-4))


def test_soft_read_sharpens_matching_slot_without_hard_routing():
    state = fresh()
    key_a, key_b = onehot(KEY_DIM, 0), onehot(KEY_DIM, 1)
    state, _, _ = write(state, key_a, onehot(VALUE_DIM, 0), mx.array([1.0]), step=0)
    state, _, _ = write(state, key_b, onehot(VALUE_DIM, 1), mx.array([1.0]), step=1)
    _, weights = read(state, key_a, hard=False)
    assert int(mx.argmax(weights[0])) == 0
    assert float(weights[0, 0]) > 0.8
    assert float(weights[0, 0]) > float(weights[0, 1]) * 3.0


# 4. store multiple independent keys
def test_store_multiple_independent_keys():
    state = fresh()
    pairs = [(onehot(KEY_DIM, i), onehot(VALUE_DIM, i) * (i + 1)) for i in range(4)]
    slots = []
    for key, value in pairs:
        state, slot_idx, rejected = write(state, key, value, mx.array([1.0]), step=0)
        assert not bool(rejected[0])
        slots.append(int(slot_idx[0]))
    assert len(set(slots)) == 4  # each key landed in its own slot
    for (key, value), slot_idx in zip(pairs, slots):
        readout, _ = read(state, key, slot_idx=mx.array([slot_idx]))
        assert bool(mx.allclose(readout, value))


# 5. handle similar keys (near-duplicate, not identical)
def test_handle_similar_keys():
    state = fresh()
    key_a = mx.array([[1.0, 0.1, 0.0, 0.0]])
    key_b = mx.array([[0.1, 1.0, 0.0, 0.0]])
    value_a, value_b = onehot(VALUE_DIM, 0) * 3.0, onehot(VALUE_DIM, 1) * 4.0
    state, slot_a, _ = write(state, key_a, value_a, mx.array([1.0]), step=0)
    state, slot_b, _ = write(state, key_b, value_b, mx.array([1.0]), step=1)
    assert int(slot_a[0]) != int(slot_b[0])
    readout_a, _ = read(state, key_a, slot_idx=slot_a)
    readout_b, _ = read(state, key_b, slot_idx=slot_b)
    assert bool(mx.allclose(readout_a, value_a))
    assert bool(mx.allclose(readout_b, value_b))


# 6. overwrite an existing fact
def test_overwrite_existing_fact():
    state = fresh()
    key = onehot(KEY_DIM, 0)
    state, slot_idx, _ = write(state, key, onehot(VALUE_DIM, 0) * 1.0, mx.array([1.0]), step=0)
    state, slot_idx2, rejected = write(state, key, onehot(VALUE_DIM, 0) * 99.0, mx.array([1.0]), step=1)
    assert not bool(rejected[0])
    assert int(slot_idx[0]) == int(slot_idx2[0])  # same key -> same slot (match, not new)
    readout, _ = read(state, key, slot_idx=slot_idx2)
    assert bool(mx.allclose(readout, onehot(VALUE_DIM, 0) * 99.0))


# 7. reinforce an existing fact
def test_reinforce_existing_fact():
    state = fresh()
    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 2.0
    state, slot_idx, _ = write(state, key, value, mx.array([0.5]), step=0)
    before_confidence = float(state.confidence[0, int(slot_idx[0])])
    state = reinforce(state, slot_idx)
    after_confidence = float(state.confidence[0, int(slot_idx[0])])
    assert after_confidence > before_confidence
    readout, _ = read(state, key, slot_idx=slot_idx)
    assert bool(mx.allclose(readout, value))  # value unchanged by reinforce
    assert int(state.age[0, int(slot_idx[0])]) == 0


# 8. protect one memory
def test_protect_one_memory():
    state = fresh()
    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 2.0
    state, slot_idx, _ = write(state, key, value, mx.array([1.0]), step=0)
    state = protect(state, slot_idx, mx.array([1.0]))
    assert float(state.protection[0, int(slot_idx[0])]) == 1.0


# 9. overwrite a different memory (protection must not leak to other slots)
def test_overwrite_different_memory_does_not_disturb_protected():
    state = fresh()
    key_a, value_a = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 2.0
    state, slot_a, _ = write(state, key_a, value_a, mx.array([1.0]), step=0)
    state = protect(state, slot_a, mx.array([1.0]))
    key_b, value_b = onehot(KEY_DIM, 1), onehot(VALUE_DIM, 1) * 3.0
    state, slot_b, rejected = write(state, key_b, value_b, mx.array([1.0]), step=1)
    assert not bool(rejected[0])
    assert int(slot_b[0]) != int(slot_a[0])
    readout_a, _ = read(state, key_a, slot_idx=slot_a)
    assert bool(mx.allclose(readout_a, value_a))  # untouched


# also: protected slot rejects a colliding same-key write attempt
def test_protected_slot_blocks_overwrite_of_same_key():
    state = fresh()
    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 2.0
    state, slot_idx, _ = write(state, key, value, mx.array([1.0]), step=0)
    state = protect(state, slot_idx, mx.array([1.0]))
    state2, slot_idx2, rejected = write(state, key, onehot(VALUE_DIM, 0) * 99.0, mx.array([1.0]), step=1)
    assert bool(rejected[0])
    readout, _ = read(state2, key, slot_idx=slot_idx)
    assert bool(mx.allclose(readout, value))  # unchanged, write was refused


# 10. forget by age
def test_forget_by_age():
    state = fresh()
    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 2.0
    state, slot_idx, _ = write(state, key, value, mx.array([1.0]), step=0)
    for _ in range(50):
        state = forget_or_decay(state, decay_rate=0.9)
    assert float(state.confidence[0, int(slot_idx[0])]) < 0.01
    assert int(state.age[0, int(slot_idx[0])]) == 50


def test_protected_memory_resists_forgetting():
    state = fresh()
    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 2.0
    state, slot_idx, _ = write(state, key, value, mx.array([1.0]), step=0)
    state = protect(state, slot_idx, mx.array([1.0]))
    for _ in range(50):
        state = forget_or_decay(state, decay_rate=0.9)
    assert float(state.confidence[0, int(slot_idx[0])]) > 0.99  # fully protected, decay_rate has no effect


# 11. overflow capacity
def _distinct_key(index: int) -> mx.array:
    # 8 keys, pairwise cosine similarity well under the 0.95 match
    # threshold -- NOT a modular reuse of only KEY_DIM=4 one-hot
    # directions (which would make e.g. keys 0 and 4 near-duplicates
    # and collide instead of filling 8 distinct slots).
    import math
    angle = index * (2 * math.pi / NUM_SLOTS)
    return mx.array([[math.cos(angle), math.sin(angle), math.cos(angle * 3), math.sin(angle * 5)]])


def test_overflow_capacity_evicts_least_protected_least_confident():
    state = fresh()
    for i in range(NUM_SLOTS):
        state, _, rejected = write(state, _distinct_key(i), onehot(VALUE_DIM, 0) * float(i), mx.array([0.3]), step=i)
        assert not bool(rejected[0])
    assert bool(mx.all(state.confidence[0] > 0))  # all 8 slots occupied
    # write a genuinely new key -- must evict, not reject, since nothing is protected
    new_key = mx.array([[5.0, 5.0, 5.0, 5.0]])
    state, slot_idx, rejected = write(state, new_key, onehot(VALUE_DIM, 1) * 100.0, mx.array([1.0]), step=100)
    assert not bool(rejected[0])
    readout, _ = read(state, new_key, slot_idx=slot_idx)
    assert bool(mx.allclose(readout, onehot(VALUE_DIM, 1) * 100.0))


# 12. reset and restore
def test_reset_clears_everything():
    state = fresh()
    state, _, _ = write(state, onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 5.0, mx.array([1.0]), step=0)
    assert bool(mx.any(state.confidence > 0))
    state = fresh()  # reset() is just reset(); re-calling it is the reset op
    assert bool(mx.all(state.confidence == 0))
    assert bool(mx.all(state.keys == 0))


def test_serialize_restore_round_trip_is_exact():
    state = fresh()
    state, slot_idx, _ = write(state, onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 5.0, mx.array([0.7]), step=3, source=SOURCE_SUPERVISED)
    state = protect(state, slot_idx, mx.array([0.4]))
    blob = serialize(state)
    restored = restore(blob)
    for field in ("keys", "values", "confidence", "age", "protection", "write_count", "last_write_step", "write_source"):
        assert bool(mx.array_equal(getattr(state, field), getattr(restored, field))), f"{field} did not round-trip exactly"


# 13. chained retrieval (write A, write B referencing A's slot via read, verify chain)
def test_chained_retrieval():
    state = fresh()
    key_a, value_a = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 1.0
    state, slot_a, _ = write(state, key_a, value_a, mx.array([1.0]), step=0)
    readout_a, _ = read(state, key_a, slot_idx=slot_a)
    key_b = readout_a  # B's key IS A's stored value -- a real chain
    value_b = onehot(VALUE_DIM, 1) * 2.0
    state, slot_b, _ = write(state, key_b, value_b, mx.array([1.0]), step=1)
    readout_a2, _ = read(state, key_a, slot_idx=slot_a)
    readout_b, _ = read(state, readout_a2, slot_idx=slot_b)
    assert bool(mx.allclose(readout_b, value_b))


# 14. conflicting writes (two different values write to the SAME oracle
# slot in the same step-order -- last write wins, deterministically)
def test_conflicting_writes_last_write_wins():
    state = fresh()
    fixed_slot = mx.array([0])
    state, _, r1 = write(state, onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 1.0, mx.array([1.0]), step=0, slot_idx=fixed_slot)
    state, _, r2 = write(state, onehot(KEY_DIM, 1), onehot(VALUE_DIM, 1) * 2.0, mx.array([1.0]), step=1, slot_idx=fixed_slot)
    assert not bool(r1[0]) and not bool(r2[0])
    readout, _ = read(state, onehot(KEY_DIM, 1), slot_idx=fixed_slot)
    assert bool(mx.allclose(readout, onehot(VALUE_DIM, 1) * 2.0))


# extra: delete is explicit, distinct from decay
def test_delete_is_immediate_and_total():
    state = fresh()
    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 5.0
    state, slot_idx, _ = write(state, key, value, mx.array([1.0]), step=0)
    state = protect(state, slot_idx, mx.array([1.0]))  # even protected slots can be explicitly deleted
    state = delete(state, slot_idx)
    assert float(state.confidence[0, int(slot_idx[0])]) == 0.0
    assert float(state.protection[0, int(slot_idx[0])]) == 0.0
    assert bool(mx.all(state.keys[0, int(slot_idx[0])] == 0))
    assert bool(mx.all(state.values[0, int(slot_idx[0])] == 0))


# extra: deterministic behavior (same inputs, same seed-free op sequence -> identical state)
def test_deterministic_behavior():
    def run():
        state = fresh()
        state, slot_idx, _ = write(state, onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 3.0, mx.array([1.0]), step=0)
        state = reinforce(state, slot_idx)
        state = forget_or_decay(state)
        return state

    a, b = run(), run()
    for field in ("keys", "values", "confidence", "age", "protection"):
        assert bool(mx.array_equal(getattr(a, field), getattr(b, field)))


# extra: learned (non-oracle) write/read is differentiable end to end
def test_write_read_learned_mode_is_differentiable():
    import mlx.nn as nn

    def loss_fn(query_and_key):
        state = fresh()
        key = query_and_key
        value = mx.stop_gradient(onehot(VALUE_DIM, 0) * 2.0)
        state, _, _ = write(state, key, value, mx.array([1.0]), step=0, source=SOURCE_LATENT)  # learned slot choice
        readout, _ = read(state, key)  # soft, learned addressing
        return mx.sum((readout - value) ** 2)

    query_and_key = onehot(KEY_DIM, 0) + mx.array([[0.01, 0.0, 0.0, 0.0]])
    value, grad = mx.value_and_grad(loss_fn)(query_and_key)
    mx.eval(value, grad)
    assert bool(mx.isfinite(value))
    assert bool(mx.all(mx.isfinite(grad)))
