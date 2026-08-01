"""HZ-0B B11: expands real task x baseline coverage by replaying B8 Stage
5's adversarial scenarios (`reference/hz0b_b8_stage5_adversarial.py`)
against all 5 B4 baselines (`reference/hz0b_baselines.py`), not just
HZ-0B's own memory simulator. No LM/training run needed -- these are pure
functional write/read sequences, same reason Stage 5 itself didn't need
one, so this covers real ground cheaply.

4 of Stage 5's 7 scenarios are meaningfully comparable across every
baseline (contradictory-info resolution, distractor immunity,
near-identical-key handling, capacity-pressure eviction) -- each
baseline exposes write(key,value)/read(query) even if the mechanism
differs wildly. The other 3 (malicious overwrite, stale memories,
stale-vs-fresh competition) are protection/confidence-specific: B4's own
baselines have no protection or confidence-decay CONCEPT at all (by
design -- "Not every baseline can do everything HZ-0B can... that
limitation is the point, not a bug to fix"). Rather than force an unfair
comparison, those 3 are reported as "N/A by construction" -- itself a
real, honest data point (these baselines cannot even represent the
scenario, let alone pass it), not a cop-out.
"""
from __future__ import annotations

import mlx.core as mx

from reference.hz0b_baselines import (
    external_retrieval_read, external_retrieval_reset, external_retrieval_write,
    large_recurrent_read, large_recurrent_reset, large_recurrent_write,
    long_context_read, long_context_reset, long_context_write,
    no_memory_read, no_memory_reset, no_memory_write,
    simple_kv_cache_read, simple_kv_cache_reset, simple_kv_cache_write,
)
from reference.hz0b_memory_simulator import protect, read as hz0b_read, reset as hz0b_reset, write as hz0b_write

KEY_DIM = VALUE_DIM = 16
NUM_SLOTS = 8


def _onehot(dim: int, index: int) -> mx.array:
    row = [1.0 if i == index else 0.0 for i in range(dim)]
    return mx.array([row])


def _close(a: mx.array, b: mx.array, atol: float = 1e-3) -> bool:
    return bool(mx.all(mx.abs(a - b) < atol))


# ---- Baseline adapters: each returns (final_readout) for a given scenario. ----

def hz0b_contradictory(key_a, fact1, fact2):
    state = hz0b_reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    state, slot, _ = hz0b_write(state, key_a, fact1, mx.array([1.0]), step=0, slot_idx=mx.array([0]))
    state, _, _ = hz0b_write(state, key_a, fact2, mx.array([1.0]), step=1, slot_idx=mx.array([0]))
    readout, _ = hz0b_read(state, key_a, hard=True)
    return readout


def no_memory_contradictory(key_a, fact1, fact2):
    state = no_memory_reset()
    state = no_memory_write(state, key_a, fact1)
    state = no_memory_write(state, key_a, fact2)
    return no_memory_read(state, key_a)


def large_recurrent_contradictory(key_a, fact1, fact2):
    state = large_recurrent_reset(1, VALUE_DIM)
    state = large_recurrent_write(state, key_a, fact1)
    state = large_recurrent_write(state, key_a, fact2)
    return large_recurrent_read(state, key_a)


def long_context_contradictory(key_a, fact1, fact2):
    state = long_context_reset(1, KEY_DIM, VALUE_DIM)
    state = long_context_write(state, key_a, fact1)
    state = long_context_write(state, key_a, fact2)
    return long_context_read(state, key_a)


def simple_kv_cache_contradictory(key_a, fact1, fact2):
    state = simple_kv_cache_reset()
    state = simple_kv_cache_write(state, key_a, fact1)
    state = simple_kv_cache_write(state, key_a, fact2)
    return simple_kv_cache_read(state, key_a)


def external_retrieval_contradictory(key_a, fact1, fact2):
    state = external_retrieval_reset(1, KEY_DIM, VALUE_DIM)
    state = external_retrieval_write(state, key_a, fact1)
    state = external_retrieval_write(state, key_a, fact2)
    return external_retrieval_read(state, key_a)


def run_scenario_contradictory():
    key_a = _onehot(KEY_DIM, 0)
    fact1, fact2 = _onehot(VALUE_DIM, 0) * 5.0, _onehot(VALUE_DIM, 1) * 5.0
    results = {
        "HZ-0B": hz0b_contradictory(key_a, fact1, fact2),
        "no-memory": no_memory_contradictory(key_a, fact1, fact2),
        "large-recurrent": large_recurrent_contradictory(key_a, fact1, fact2),
        "long-context": long_context_contradictory(key_a, fact1, fact2),
        "simple-kv-cache": simple_kv_cache_contradictory(key_a, fact1, fact2),
        "external-retrieval": external_retrieval_contradictory(key_a, fact1, fact2),
    }
    # correct behavior: returns fact2 (the LATEST write), not fact1, not a blend
    return {name: _close(readout, fact2) and not _close(readout, fact1) for name, readout in results.items()}


def run_scenario_distractors():
    real_key, real_value = _onehot(KEY_DIM, 0), _onehot(VALUE_DIM, 0) * 5.0
    distractors = [(_onehot(KEY_DIM, i), _onehot(VALUE_DIM, i) * 3.0) for i in range(1, 6)]

    def replay(reset_fn, write_fn, read_fn, *reset_args):
        state = reset_fn(*reset_args)
        state = write_fn(state, real_key, real_value)
        for dk, dv in distractors:
            state = write_fn(state, dk, dv)
        return read_fn(state, real_key)

    hz0b_state = hz0b_reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    hz0b_state, _, _ = hz0b_write(hz0b_state, real_key, real_value, mx.array([1.0]), step=0, slot_idx=mx.array([0]))
    for i, (dk, dv) in enumerate(distractors):
        hz0b_state, _, _ = hz0b_write(hz0b_state, dk, dv, mx.array([1.0]), step=i + 1, slot_idx=mx.array([i + 1]))
    hz0b_readout, _ = hz0b_read(hz0b_state, real_key, hard=True)

    results = {
        "HZ-0B": hz0b_readout,
        "no-memory": replay(no_memory_reset, no_memory_write, no_memory_read),
        "large-recurrent": replay(large_recurrent_reset, large_recurrent_write, large_recurrent_read, 1, VALUE_DIM),
        "long-context": replay(long_context_reset, long_context_write, long_context_read, 1, KEY_DIM, VALUE_DIM),
        "simple-kv-cache": replay(simple_kv_cache_reset, simple_kv_cache_write, simple_kv_cache_read),
        "external-retrieval": replay(external_retrieval_reset, external_retrieval_write, external_retrieval_read, 1, KEY_DIM, VALUE_DIM),
    }
    return {name: _close(readout, real_value) for name, readout in results.items()}


def run_scenario_near_identical_keys():
    key_a = _onehot(KEY_DIM, 0)
    key_b_raw = 0.995 * key_a + (1 - 0.995 ** 2) ** 0.5 * _onehot(KEY_DIM, 1)
    key_b = key_b_raw / mx.sqrt(mx.sum(key_b_raw * key_b_raw))
    value_a, value_b = _onehot(VALUE_DIM, 0) * 5.0, _onehot(VALUE_DIM, 1) * 5.0

    def replay(reset_fn, write_fn, read_fn, *reset_args):
        state = reset_fn(*reset_args)
        state = write_fn(state, key_a, value_a)
        state = write_fn(state, key_b, value_b)
        return read_fn(state, key_a), read_fn(state, key_b)

    state = hz0b_reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    state, slot_a, _ = hz0b_write(state, key_a, value_a, mx.array([1.0]), step=0, slot_idx=mx.array([0]))
    state, slot_b, _ = hz0b_write(state, key_b, value_b, mx.array([1.0]), step=1)
    hz0b_ra, _ = hz0b_read(state, key_a, hard=True)
    hz0b_rb, _ = hz0b_read(state, key_b, hard=True)

    others = {
        "no-memory": replay(no_memory_reset, no_memory_write, no_memory_read),
        "large-recurrent": replay(large_recurrent_reset, large_recurrent_write, large_recurrent_read, 1, VALUE_DIM),
        "long-context": replay(long_context_reset, long_context_write, long_context_read, 1, KEY_DIM, VALUE_DIM),
        "simple-kv-cache": replay(simple_kv_cache_reset, simple_kv_cache_write, simple_kv_cache_read),
        "external-retrieval": replay(external_retrieval_reset, external_retrieval_write, external_retrieval_read, 1, KEY_DIM, VALUE_DIM),
    }
    results = {"HZ-0B": (hz0b_ra, hz0b_rb), **others}
    # correct behavior: both facts stay distinct and retrievable
    return {name: _close(ra, value_a) and _close(rb, value_b) for name, (ra, rb) in results.items()}


def run_scenario_capacity_pressure(num_facts: int = 12):
    facts = [(_onehot(KEY_DIM, i % KEY_DIM), _onehot(VALUE_DIM, i % VALUE_DIM) * 2.0) for i in range(num_facts)]
    first_key, first_value = facts[0]

    def replay(reset_fn, write_fn, read_fn, *reset_args):
        state = reset_fn(*reset_args)
        for k, v in facts:
            state = write_fn(state, k, v)
        return read_fn(state, first_key)

    # HZ-0B: no protection here (capacity pressure WITHOUT the protection escape
    # hatch, unlike Stage 5's own version) -- a fair, apples-to-apples eviction test.
    state = hz0b_reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    for i, (k, v) in enumerate(facts):
        state, _, _ = hz0b_write(state, k, v, mx.array([1.0]), step=i)
    hz0b_readout, _ = hz0b_read(state, first_key, hard=True)

    results = {
        "HZ-0B": hz0b_readout,
        "no-memory": replay(no_memory_reset, no_memory_write, no_memory_read),
        "large-recurrent": replay(large_recurrent_reset, large_recurrent_write, large_recurrent_read, 1, VALUE_DIM),
        "long-context": replay(long_context_reset, long_context_write, long_context_read, 1, KEY_DIM, VALUE_DIM),
        "simple-kv-cache": replay(simple_kv_cache_reset, simple_kv_cache_write, simple_kv_cache_read),
        "external-retrieval": replay(external_retrieval_reset, external_retrieval_write, external_retrieval_read, 1, KEY_DIM, VALUE_DIM),
    }
    # "pass" here means the FIRST fact (num_facts-1 writes ago, unprotected,
    # NUM_SLOTS=8 < num_facts=12) is still exactly retrievable -- an honest
    # bar, since HZ-0B itself may legitimately evict it without protection.
    return {name: _close(readout, first_value) for name, readout in results.items()}


def run_scenario_noisy_query():
    """The real differentiator the other 4 scenarios don't exercise:
    read with a NOISY query (same direction, small perturbation), not
    the exact key used at write time. `simple-kv-cache`'s exact hash
    lookup fails outright on any noisy query by construction -- this is
    what actually distinguishes similarity-based content addressing
    (HZ-0B, external-retrieval's own nearest-neighbor, long-context's
    soft attention) from exact-match lookup, which the other 4 scenarios
    (all read with the literal write-time key) don't test at all."""
    key = _onehot(KEY_DIM, 0)
    value = _onehot(VALUE_DIM, 0) * 5.0
    noise = mx.array([[0.05 if i == 1 else 0.0 for i in range(KEY_DIM)]])
    noisy_query = key + noise
    noisy_query = noisy_query / mx.sqrt(mx.sum(noisy_query * noisy_query))

    def replay(reset_fn, write_fn, read_fn, *reset_args):
        state = reset_fn(*reset_args)
        state = write_fn(state, key, value)
        return read_fn(state, noisy_query)

    state = hz0b_reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    state, _, _ = hz0b_write(state, key, value, mx.array([1.0]), step=0, slot_idx=mx.array([0]))
    hz0b_readout, _ = hz0b_read(state, noisy_query, hard=True)

    results = {
        "HZ-0B": hz0b_readout,
        "no-memory": replay(no_memory_reset, no_memory_write, no_memory_read),
        "large-recurrent": replay(large_recurrent_reset, large_recurrent_write, large_recurrent_read, 1, VALUE_DIM),
        "long-context": replay(long_context_reset, long_context_write, long_context_read, 1, KEY_DIM, VALUE_DIM),
        "simple-kv-cache": replay(simple_kv_cache_reset, simple_kv_cache_write, simple_kv_cache_read),
        "external-retrieval": replay(external_retrieval_reset, external_retrieval_write, external_retrieval_read, 1, KEY_DIM, VALUE_DIM),
    }
    return {name: _close(readout, value, atol=0.5) for name, readout in results.items()}


def run_scenario_noisy_query_with_distractors():
    """Fixes a real confound in scenario 5: with only ONE item ever
    stored, `large-recurrent`'s content-blind accumulator and
    `long-context`'s softmax attention both trivially return the only
    thing they have regardless of query quality -- a vacuous "pass" that
    doesn't test real noisy-query robustness (same root issue as the B11
    factorial diagnosis's cell-1 confound: a 1-item memory trivially
    "solves" single-item retrieval no matter how the read mechanism
    works). This scenario combines scenario 2's distractors with
    scenario 5's noisy query so a genuine query-dependent selection is
    actually required: with 6 stored items, a content-blind blend or a
    softmax that can't tell the query apart from the wrong keys will
    return a mixture/wrong answer, not the real fact."""
    real_key, real_value = _onehot(KEY_DIM, 0), _onehot(VALUE_DIM, 0) * 5.0
    distractors = [(_onehot(KEY_DIM, i), _onehot(VALUE_DIM, i) * 3.0) for i in range(1, 6)]
    noise = mx.array([[0.05 if i == 6 else 0.0 for i in range(KEY_DIM)]])
    noisy_query = real_key + noise
    noisy_query = noisy_query / mx.sqrt(mx.sum(noisy_query * noisy_query))

    def replay(reset_fn, write_fn, read_fn, *reset_args):
        state = reset_fn(*reset_args)
        state = write_fn(state, real_key, real_value)
        for dk, dv in distractors:
            state = write_fn(state, dk, dv)
        return read_fn(state, noisy_query)

    hz0b_state = hz0b_reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    hz0b_state, _, _ = hz0b_write(hz0b_state, real_key, real_value, mx.array([1.0]), step=0, slot_idx=mx.array([0]))
    for i, (dk, dv) in enumerate(distractors):
        hz0b_state, _, _ = hz0b_write(hz0b_state, dk, dv, mx.array([1.0]), step=i + 1, slot_idx=mx.array([i + 1]))
    hz0b_readout, _ = hz0b_read(hz0b_state, noisy_query, hard=True)

    results = {
        "HZ-0B": hz0b_readout,
        "no-memory": replay(no_memory_reset, no_memory_write, no_memory_read),
        "large-recurrent": replay(large_recurrent_reset, large_recurrent_write, large_recurrent_read, 1, VALUE_DIM),
        "long-context": replay(long_context_reset, long_context_write, long_context_read, 1, KEY_DIM, VALUE_DIM),
        "simple-kv-cache": replay(simple_kv_cache_reset, simple_kv_cache_write, simple_kv_cache_read),
        "external-retrieval": replay(external_retrieval_reset, external_retrieval_write, external_retrieval_read, 1, KEY_DIM, VALUE_DIM),
    }
    return {name: _close(readout, real_value, atol=0.5) for name, readout in results.items()}


def main():
    print(f"Capacity-comparable adversarial scenarios, HZ-0B vs. all 5 B4 baselines (KEY_DIM=VALUE_DIM={KEY_DIM}, NUM_SLOTS={NUM_SLOTS}):\n")
    scenarios = {
        "1. Contradictory later information (must return LATEST write)": run_scenario_contradictory(),
        "2. Distractor immunity (real fact survives 5 interspersed distractors)": run_scenario_distractors(),
        "3. Near-identical keys (two different facts must stay distinct)": run_scenario_near_identical_keys(),
        "4. Capacity pressure, no protection (first fact after 11 more writes, 8 slots)": run_scenario_capacity_pressure(),
        "5. Noisy query, single item (vacuous for content-blind/single-item baselines -- see caveat below)": run_scenario_noisy_query(),
        "6. Noisy query WITH 5 distractors (the real, non-confounded version of 5)": run_scenario_noisy_query_with_distractors(),
    }
    conditions = ["HZ-0B", "no-memory", "large-recurrent", "long-context", "simple-kv-cache", "external-retrieval"]
    header = f"{'Scenario':<70}" + "".join(f"{c:>18}" for c in conditions)
    print(header)
    for name, result in scenarios.items():
        row = f"{name:<70}" + "".join(f"{'PASS' if result[c] else 'fail':>18}" for c in conditions)
        print(row)

    print("\nCaveat on scenario 5: with only ONE item ever stored, large-recurrent's")
    print("content-blind accumulator and long-context's softmax attention trivially")
    print("return the only thing they have regardless of query quality -- a vacuous")
    print("pass, not real noisy-query robustness. Scenario 6 fixes this (distractors +")
    print("noisy query) and is the one that actually matters; 5 is kept only to show")
    print("the confound explicitly rather than silently skip it.")

    print("\nNot run (N/A by construction, not a cop-out -- these baselines have no")
    print("protection or confidence-decay CONCEPT at all, per reference/hz0b_baselines.py's")
    print("own design): malicious overwrite attempt, stale memories, stale-vs-fresh")
    print("competition. HZ-0B is the only condition capable of representing these")
    print("scenarios at all -- see docs/restart/hz0b_b8_stage5_results.md for HZ-0B's")
    print("own real results on all 7 scenarios (5/7 correct out of the box, 2 real bugs")
    print("found and fixed same-day).")

    print("\n--- Summary ---")
    for c in conditions:
        passed = sum(1 for r in scenarios.values() if r[c])
        print(f"{c:<20}: {passed}/{len(scenarios)} scenarios passed")


if __name__ == "__main__":
    main()
