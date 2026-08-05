"""Generate a cross-language parity fixture for
`restart/hz0a_pmetal/crates/hz0d-pmetal-fastweights`: runs a real,
meaningful sequence of D1 fast-weight operations through the actual
Python reference (`reference/hz0d_fast_weights.py`) for TWO INDEPENDENT
single-session traces (different seeds/gradients), and dumps every
input and the resulting state as JSON. The Rust port's own test
(`tests/parity.rs`) replays BOTH traces through ONE batched
(`sessions=2`) call and asserts: (a) each session's final state matches
its own independently-computed Python trace exactly, proving the batch
dimension does not leak between sessions, and (b) `apply()` on the
batched state reproduces each session's individually-computed output --
the same "strongest form of agreement" pattern
`scripts/hz0b_generate_rust_parity_fixture.py` already established.
"""
from __future__ import annotations

import json

import mlx.core as mx

from reference.hz0d_fast_weights import (
    FastWeightConfig, FastWeightState, apply_fast_linear, decay_fast_weights, effective_delta,
    init_fast_weights, update_fast_weights,
)

DIM, RANK, NUM_LAYERS = 8, 2, 2
CONFIG = FastWeightConfig(dim=DIM, rank=RANK, num_layers=NUM_LAYERS, max_delta_norm=1.0)


def to_list(arr):
    return arr.tolist() if hasattr(arr, "tolist") else arr


def run_one_session_trace(*, init_seed: int, grad_seed: int, decay_rate: float):
    state = init_fast_weights(FastWeightConfig(dim=DIM, rank=RANK, num_layers=NUM_LAYERS, max_delta_norm=1.0, init_seed=init_seed))
    trace = {"a_fast_init": to_list(state.a_fast), "steps": []}

    key = mx.random.key(grad_seed)
    for layer in range(NUM_LAYERS):
        k1, k2, key = mx.random.split(key, 3)
        grad_a = mx.random.normal((DIM, RANK), key=k1) * 0.5
        grad_b = mx.random.normal((RANK, DIM), key=k2) * 0.5
        state = update_fast_weights(state, layer, grad_a, grad_b, lr=0.1, config=CONFIG)
        trace["steps"].append({"op": "update", "layer": layer, "grad_a": to_list(grad_a), "grad_b": to_list(grad_b), "lr": 0.1, "max_delta_norm": 1.0})

    state = decay_fast_weights(state, decay_rate)
    trace["steps"].append({"op": "decay", "decay_rate": decay_rate})

    key_x, _ = mx.random.split(key)
    x = mx.random.normal((1, DIM), key=key_x) * 0.3
    base_weight = mx.random.normal((DIM, DIM), key=key_x) * 0.05
    base_bias = mx.random.normal((DIM,), key=key_x) * 0.01
    y = apply_fast_linear(x, base_weight, base_bias, state, 0)
    delta0 = effective_delta(state, 0)
    delta1 = effective_delta(state, 1)

    trace["apply"] = {"x": to_list(x), "base_weight": to_list(base_weight), "base_bias": to_list(base_bias), "layer": 0, "y": to_list(y)}
    trace["final_state"] = {"a_fast": to_list(state.a_fast), "b_fast": to_list(state.b_fast), "update_count": int(state.update_count)}
    trace["final_delta_layer0"] = to_list(delta0)
    trace["final_delta_layer1"] = to_list(delta1)
    return trace


def main():
    fixture = {
        "dim": DIM, "rank": RANK, "num_layers": NUM_LAYERS,
        "session0": run_one_session_trace(init_seed=1, grad_seed=100, decay_rate=0.9),
        "session1": run_one_session_trace(init_seed=2, grad_seed=200, decay_rate=1.0),
    }
    out_path = "restart/hz0a_pmetal/crates/hz0d-pmetal-fastweights/tests/fixture.json"
    with open(out_path, "w") as f:
        json.dump(fixture, f)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
