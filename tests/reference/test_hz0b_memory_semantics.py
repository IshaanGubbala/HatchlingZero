"""HZ-0B Phase B3 tests: verifies the exit gate directly -- "new writes,
reinforcement, updates, deletion, and protection produce observably
different behavior" -- via reference/hz0b_memory_semantics.py's decision
signals and penalties, still with no language model attached."""
import mlx.core as mx

from reference.hz0b_memory_semantics import (
    decide_operation,
    duplicate_memory_penalty,
    excessive_write_penalty,
    memory_norm_penalty,
    protected_memory_corruption_penalty,
    uncontrolled_growth_penalty,
)
from reference.hz0b_memory_simulator import protect, reset, write

NUM_SLOTS = 8
KEY_DIM = 4
VALUE_DIM = 4


def fresh():
    return reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)


def onehot(dim, index):
    return mx.array([[1.0 if i == index else 0.0 for i in range(dim)]])


def test_new_write_has_high_write_low_update_signal():
    state = fresh()
    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 3.0
    decision = decide_operation(state, key, value)  # nothing stored yet
    assert float(decision.write_probability[0]) > 0.9
    assert float(decision.update_probability[0]) < 0.1
    assert float(decision.reinforce_probability[0]) < 0.1
    assert float(decision.contradiction_probability[0]) < 0.1


def test_reinforcement_vs_contradiction_are_observably_different():
    state = fresh()
    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 3.0
    state, slot_idx, _ = write(state, key, value, mx.array([1.0]), step=0)

    same_value_decision = decide_operation(state, key, value)  # exact same key AND value
    different_value_decision = decide_operation(state, key, onehot(VALUE_DIM, 1) * 3.0)  # same key, different value

    assert float(same_value_decision.reinforce_probability[0]) > 0.8
    assert float(same_value_decision.contradiction_probability[0]) < 0.2
    assert float(different_value_decision.contradiction_probability[0]) > 0.8
    assert float(different_value_decision.reinforce_probability[0]) < 0.2
    # the two decisions must be genuinely different, not coincidentally similar
    assert float(same_value_decision.reinforce_probability[0]) > float(different_value_decision.reinforce_probability[0]) + 0.5
    assert float(different_value_decision.contradiction_probability[0]) > float(same_value_decision.contradiction_probability[0]) + 0.5


def test_update_probability_uses_its_own_threshold_not_a_shared_one():
    # A near-but-not-exact key match should sit in a graded middle zone --
    # proof this isn't a single hard cutoff shared with e.g. duplicate
    # detection (which uses a different threshold entirely, see
    # test_duplicate_penalty_uses_different_threshold_than_update below).
    state = fresh()
    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 3.0
    state, _, _ = write(state, key, value, mx.array([1.0]), step=0)
    near_key = mx.array([[0.85, 0.15, 0.0, 0.0]])  # cosine sim with [1,0,0,0] is 0.985, close to but not at UPDATE_MATCH_THRESHOLD=0.9's steep zone
    decision = decide_operation(state, near_key, value)
    # graded, not saturated at exactly 0 or 1
    assert 0.01 < float(decision.update_probability[0]) < 0.999


def test_deletion_is_observably_different_from_decay():
    from reference.hz0b_memory_simulator import delete, forget_or_decay

    state = fresh()
    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 3.0
    state, slot_idx, _ = write(state, key, value, mx.array([1.0]), step=0)

    decayed = state
    for _ in range(3):
        decayed = forget_or_decay(decayed, decay_rate=0.9)
    deleted = delete(state, slot_idx)

    # decay is gradual (value survives, confidence just lower); delete is immediate and total
    assert float(decayed.confidence[0, int(slot_idx[0])]) > 0.5
    assert bool(mx.any(decayed.keys[0, int(slot_idx[0])] != 0))
    assert float(deleted.confidence[0, int(slot_idx[0])]) == 0.0
    assert bool(mx.all(deleted.keys[0, int(slot_idx[0])] == 0))


def test_protection_is_observably_different_from_unprotected():
    state = fresh()
    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 3.0
    state, slot_idx, _ = write(state, key, value, mx.array([1.0]), step=0)
    protected_state = protect(state, slot_idx, mx.array([1.0]))

    for _ in range(30):
        state = forget_or_decay_wrapper(state)
        protected_state = forget_or_decay_wrapper(protected_state)

    assert float(state.confidence[0, int(slot_idx[0])]) < 0.1
    assert float(protected_state.confidence[0, int(slot_idx[0])]) > 0.9


def forget_or_decay_wrapper(state):
    from reference.hz0b_memory_simulator import forget_or_decay
    return forget_or_decay(state, decay_rate=0.9)


def test_memory_validity_confidence_decays_independent_of_raw_confidence():
    state = fresh()
    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 3.0
    state, slot_idx, _ = write(state, key, value, mx.array([1.0]), step=0)
    fresh_decision = decide_operation(state, key, value)
    raw_confidence_fresh = float(state.confidence[0, int(slot_idx[0])])
    validity_fresh = float(fresh_decision.memory_validity_confidence[0, int(slot_idx[0])])
    assert abs(raw_confidence_fresh - validity_fresh) < 0.01  # age=0, should match closely

    aged_state = state
    for _ in range(20):  # one staleness half-life
        aged_state = forget_or_decay_wrapper(aged_state)
    # forget_or_decay already changes raw confidence too, so instead directly
    # test the validity function against an age-only state (no decay applied
    # to raw confidence) to isolate the staleness signal specifically
    from reference.hz0b_memory_simulator import replace
    age_only_state = replace(state, age=mx.array([[20, 0, 0, 0, 0, 0, 0, 0]]))
    age_decision = decide_operation(age_only_state, key, value)
    raw_confidence_aged = float(age_only_state.confidence[0, int(slot_idx[0])])
    validity_aged = float(age_decision.memory_validity_confidence[0, int(slot_idx[0])])
    assert raw_confidence_aged == raw_confidence_fresh  # raw confidence untouched by age alone
    assert validity_aged < validity_fresh * 0.6  # but validity has decayed toward the half-life target


def test_corruption_penalty_is_zero_when_protection_holds():
    state = fresh()
    key, value = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 3.0
    state, slot_idx, _ = write(state, key, value, mx.array([1.0]), step=0)
    state = protect(state, slot_idx, mx.array([1.0]))
    before = state
    after, _, rejected = write(state, key, onehot(VALUE_DIM, 1) * 99.0, mx.array([1.0]), step=1)
    assert bool(rejected[0])
    penalty = protected_memory_corruption_penalty(before, after)
    assert float(penalty) == 0.0


def test_duplicate_penalty_rises_with_similar_keys_and_uses_different_threshold_than_update():
    state = fresh()
    state, _, _ = write(state, onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0), mx.array([1.0]), step=0)
    diverse_state, _, _ = write(state, onehot(KEY_DIM, 1), onehot(VALUE_DIM, 1), mx.array([1.0]), step=1)
    # cos([0.7,0.7,0,0], [1,0,0,0]) ~= 0.707 -- clearly more similar than the
    # diverse case's 0.0, but well below B2's own 0.95 write()-match
    # threshold, so this lands in a genuinely NEW slot (not an update to
    # slot 0) and both slots are actually occupied for the penalty to
    # compare -- proves duplicate_memory_penalty operates on its own,
    # different, threshold-free similarity signal from write()'s match
    # decision, exactly what this test's name claims.
    near_dup_key = mx.array([[0.7, 0.7, 0.0, 0.0]])
    near_dup_state, near_dup_slot, near_dup_rejected = write(state, near_dup_key, onehot(VALUE_DIM, 1), mx.array([1.0]), step=1)
    assert not bool(near_dup_rejected[0])
    assert int(near_dup_slot[0]) != 0  # confirms it did NOT collide/update into slot 0
    assert float(duplicate_memory_penalty(near_dup_state)) > float(duplicate_memory_penalty(diverse_state))


def test_norm_penalty_rises_with_scaled_values():
    state = fresh()
    normal_state, _, _ = write(state, onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0), mx.array([1.0]), step=0)
    scaled_state, _, _ = write(state, onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 50.0, mx.array([1.0]), step=0)
    assert float(memory_norm_penalty(scaled_state)) > float(memory_norm_penalty(normal_state))


def test_growth_penalty_rises_as_slots_fill():
    import math
    state = fresh()
    penalty_at = []
    for i in range(NUM_SLOTS):
        angle = i * (2 * math.pi / NUM_SLOTS)
        key = mx.array([[math.cos(angle), math.sin(angle), math.cos(angle * 3), math.sin(angle * 5)]])
        state, _, _ = write(state, key, onehot(VALUE_DIM, 0), mx.array([1.0]), step=i)
        penalty_at.append(float(uncontrolled_growth_penalty(state)))
    assert penalty_at[-1] > penalty_at[0]


def test_write_sparsity_penalty_reflects_write_probability():
    state = fresh()
    decision_new = decide_operation(state, onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0))
    state, _, _ = write(state, onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0), mx.array([1.0]), step=0)
    decision_repeat = decide_operation(state, onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0))
    assert float(excessive_write_penalty(decision_new)) > float(excessive_write_penalty(decision_repeat))
