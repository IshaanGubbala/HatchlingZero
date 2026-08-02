"""HZ-0B B11: reinforcement / forgetting / serialization accuracy.

One of B11's 16 named eval tasks, still unstarted. Unlike the earlier
correctness unit tests for `reinforce`/`forget_or_decay`/`serialize`/
`restore` (tests/reference/test_hz0b_memory_simulator.py etc, which
check the mechanism's local behavior in isolation), this measures real
RETRIEVAL ACCURACY at the end of an extended session: do reinforced
facts stay retrievable while unreinforced facts decay into
unretrievability, and does a serialize/restore roundtrip preserve that
retrieval behavior EXACTLY. Pure B2 simulator (no LM needed -- these
are properties of the memory mechanism itself, same reasoning as B8
Stage 5 and the reopening-criteria scripts).
"""
from __future__ import annotations

import argparse

import mlx.core as mx

from reference.hz0b_memory_simulator import (
    forget_or_decay, read, reinforce, reset, restore, serialize, write,
)

NUM_SLOTS = 8
KEY_DIM = VALUE_DIM = 32
NUM_FACTS = 6
REINFORCED_IDX = [0, 1, 2]
UNREINFORCED_IDX = [3, 4, 5]
SEED = 555


def retrieval_accuracy(state, keys) -> dict[int, bool]:
    results = {}
    for i in range(NUM_FACTS):
        _, weights = read(state, keys[i:i + 1], hard=True, confidence_weighted=True)
        predicted_slot = int(mx.argmax(weights[0]))
        results[i] = predicted_slot == i
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--reinforce-every", type=int, default=10)
    parser.add_argument("--decay-rate", type=float, default=0.85)
    args = parser.parse_args()
    STEPS, REINFORCE_EVERY, DECAY_RATE = args.steps, args.reinforce_every, args.decay_rate

    mx.random.seed(SEED)
    state = reset(batch_size=1, num_slots=NUM_SLOTS, key_dim=KEY_DIM, value_dim=VALUE_DIM)
    keys = mx.random.normal((NUM_FACTS, KEY_DIM))
    values = mx.random.normal((NUM_FACTS, VALUE_DIM))

    for i in range(NUM_FACTS):
        state, _, rejected = write(state, keys[i:i + 1], values[i:i + 1], mx.array([1.0]), step=0, slot_idx=mx.array([i]))
        assert not bool(rejected[0]), f"fact {i} write unexpectedly rejected"

    print(f"Wrote {NUM_FACTS} facts into {NUM_SLOTS} slots (oracle slot assignment, no eviction confound).")
    print(f"Simulating {STEPS} decay steps (decay_rate={DECAY_RATE}), reinforcing slots {REINFORCED_IDX} every {REINFORCE_EVERY} steps, slots {UNREINFORCED_IDX} left to decay naturally.\n")

    confidence_trace = {i: [] for i in range(NUM_FACTS)}
    for step in range(1, STEPS + 1):
        state = forget_or_decay(state, decay_rate=DECAY_RATE)
        if step % REINFORCE_EVERY == 0:
            for idx in REINFORCED_IDX:
                state = reinforce(state, slot_idx=mx.array([idx]))
        for i in range(NUM_FACTS):
            confidence_trace[i].append(float(state.confidence[0, i]))

    print("Final confidence per fact:")
    for i in range(NUM_FACTS):
        tag = "reinforced" if i in REINFORCED_IDX else "unreinforced"
        print(f"  fact {i} ({tag}): confidence {float(state.confidence[0, i]):.4f}")

    acc = retrieval_accuracy(state, keys)
    reinforced_acc = sum(acc[i] for i in REINFORCED_IDX) / len(REINFORCED_IDX)
    unreinforced_acc = sum(acc[i] for i in UNREINFORCED_IDX) / len(UNREINFORCED_IDX)
    print(f"\nRetrieval accuracy at step {STEPS}:")
    print(f"  reinforced facts:   {reinforced_acc:.3f}  (per-fact: {[acc[i] for i in REINFORCED_IDX]})")
    print(f"  unreinforced facts: {unreinforced_acc:.3f}  (per-fact: {[acc[i] for i in UNREINFORCED_IDX]})")

    print("\nSerialize/restore roundtrip:")
    blob = serialize(state)
    restored = restore(blob)
    fields_match = all(
        bool(mx.array_equal(getattr(state, f), getattr(restored, f)))
        for f in ("keys", "values", "confidence", "age", "protection", "write_count", "last_write_step", "write_source")
    )
    print(f"  all 8 fields bit-exact after restore: {fields_match}")

    orig_acc = retrieval_accuracy(state, keys)
    restored_acc = retrieval_accuracy(restored, keys)
    accuracy_preserved = orig_acc == restored_acc
    print(f"  retrieval accuracy identical before/after restore: {accuracy_preserved}  (orig={orig_acc}, restored={restored_acc})")

    readouts_match = True
    for i in range(NUM_FACTS):
        r_orig, _ = read(state, keys[i:i + 1], hard=True, confidence_weighted=True)
        r_restored, _ = read(restored, keys[i:i + 1], hard=True, confidence_weighted=True)
        if not bool(mx.array_equal(r_orig, r_restored)):
            readouts_match = False
    print(f"  readout values bit-exact before/after restore: {readouts_match}")


if __name__ == "__main__":
    main()
