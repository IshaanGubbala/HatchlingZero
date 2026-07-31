"""Generate a cross-language parity fixture for
`restart/hz0a_pmetal/crates/hz0b-pmetal-memory`: runs a real, meaningful
sequence of B2 memory operations through the actual Python reference
(`reference/hz0b_memory_simulator.py`, as fixed 2026-07-30) and dumps
every input and the resulting state/readouts as JSON, for the Rust port's
own test (`tests/parity.rs`) to replay the identical sequence and assert
exact agreement -- the same "strongest form of agreement" pattern
`scripts/hz0a_generate_rust_parity_fixture.py` already established in
this project for HZ-0A's own PMetal port.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from reference.hz0b_memory_simulator import forget_or_decay, protect, read, reset, update, write

NUM_SLOTS, KEY_DIM, VALUE_DIM = 8, 4, 4


def onehot(dim, index):
    return mx.array([[1.0 if i == index else 0.0 for i in range(dim)]])


def to_list(arr):
    return arr.tolist() if hasattr(arr, "tolist") else arr


def main():
    fixture = {"num_slots": NUM_SLOTS, "key_dim": KEY_DIM, "value_dim": VALUE_DIM, "steps": []}

    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    key_a, value_a = onehot(KEY_DIM, 0), onehot(VALUE_DIM, 0) * 3.0
    key_b, value_b = onehot(KEY_DIM, 1), onehot(VALUE_DIM, 1) * 4.0

    state, slot_a, rejected_a = write(state, key_a, value_a, mx.array([1.0]), step=0, source=1)
    fixture["steps"].append({"op": "write", "key": to_list(key_a), "value": to_list(value_a), "strength": 1.0, "step": 0, "source": 1, "slot_idx": None, "result_slot": to_list(slot_a), "rejected": to_list(rejected_a)})

    state, slot_b, rejected_b = write(state, key_b, value_b, mx.array([1.0]), step=1, source=1)
    fixture["steps"].append({"op": "write", "key": to_list(key_b), "value": to_list(value_b), "strength": 1.0, "step": 1, "source": 1, "slot_idx": None, "result_slot": to_list(slot_b), "rejected": to_list(rejected_b)})

    readout_a, weights_a = read(state, key_a, hard=True)
    fixture["steps"].append({"op": "read", "query": to_list(key_a), "slot_idx": None, "hard": True, "confidence_weighted": True, "readout": to_list(readout_a), "weights": to_list(weights_a)})

    state = protect(state, slot_b, mx.array([1.0]))
    fixture["steps"].append({"op": "protect", "slot_idx": to_list(slot_b), "strength": 1.0})

    attacker_value = onehot(VALUE_DIM, 2) * 99.0
    state, _, rejected_attack = write(state, key_b, attacker_value, mx.array([1.0]), step=2, source=1, slot_idx=slot_b)
    fixture["steps"].append({"op": "write", "key": to_list(key_b), "value": to_list(attacker_value), "strength": 1.0, "step": 2, "source": 1, "slot_idx": to_list(slot_b), "result_slot": to_list(slot_b), "rejected": to_list(rejected_attack)})

    new_value_a = onehot(VALUE_DIM, 3) * 7.0
    state = update(state, slot_a, new_value_a)
    fixture["steps"].append({"op": "update", "slot_idx": to_list(slot_a), "new_value": to_list(new_value_a)})

    for i in range(5):
        state = forget_or_decay(state, decay_rate=0.9)
        fixture["steps"].append({"op": "forget_or_decay", "decay_rate": 0.9})

    final_readout_a, _ = read(state, key_a, hard=True)
    final_readout_b, _ = read(state, key_b, hard=True)

    fixture["final_state"] = {
        "keys": to_list(state.keys), "values": to_list(state.values), "confidence": to_list(state.confidence),
        "age": to_list(state.age), "protection": to_list(state.protection), "write_count": to_list(state.write_count),
        "last_write_step": to_list(state.last_write_step), "write_source": to_list(state.write_source),
    }
    fixture["final_readout_a"] = to_list(final_readout_a)
    fixture["final_readout_b"] = to_list(final_readout_b)

    output_path = Path("restart/hz0a_pmetal/crates/hz0b-pmetal-memory/tests/fixture.json")
    output_path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(f"wrote {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
