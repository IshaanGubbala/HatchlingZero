"""HZ-0D D9: real Python<->Rust FFI bridge correctness
(`restart/hz0a_pmetal/python/hz0d_fastweights_bridge.py`).

Two independent checks, matching this project's established
`test_hz0b_memory_rust_bridge.py` convention:
1. Replays the SAME fixture `tests/parity.rs` already uses through the
   ctypes bridge -- proving the FULL round trip (Python -> ctypes ->
   Rust -> ctypes -> Python), not just the Rust-internal replay.
2. A live comparison against `reference/hz0d_fast_weights.py` on a
   FRESH sequence generated in this test process (not the frozen
   fixture), catching any drift a stale fixture could hide.

Skips (not fails) if the bridge's cdylib hasn't been built.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

_BRIDGE_PYTHON_DIR = Path(__file__).resolve().parents[2] / "restart" / "hz0a_pmetal" / "python"
_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "restart" / "hz0a_pmetal" / "crates" / "hz0d-pmetal-fastweights" / "tests" / "fixture.json"

sys.path.insert(0, str(_BRIDGE_PYTHON_DIR))

try:
    import hz0d_fastweights_bridge as bridge
    _BRIDGE_AVAILABLE = True
except (FileNotFoundError, OSError):
    _BRIDGE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _BRIDGE_AVAILABLE or not _FIXTURE_PATH.exists(),
    reason="hz0d-pmetal-fastweights-bridge cdylib or fixture not present (cargo build --release -p "
           "hz0d-pmetal-fastweights-bridge; PYTHONPATH=. python scripts/hz0d_generate_rust_parity_fixture.py)",
)


def _run_session_through_bridge(trace: dict, *, session: int, sessions: int, dim: int, rank: int, num_layers: int, a_init: np.ndarray):
    state = bridge.reset(sessions, num_layers, dim, rank, a_init)
    for step in trace["steps"]:
        if step["op"] != "update":
            continue
        grad_a = np.array(step["grad_a"], dtype=np.float32)
        grad_b = np.array(step["grad_b"], dtype=np.float32)
        state = bridge.update(state, session, step["layer"], grad_a, grad_b, lr=step["lr"], max_delta_norm=step["max_delta_norm"])
    return state


def test_bridge_full_round_trip_matches_the_python_reference_fixture():
    fixture = json.loads(_FIXTURE_PATH.read_text())
    dim, rank, num_layers = fixture["dim"], fixture["rank"], fixture["num_layers"]
    session0, session1 = fixture["session0"], fixture["session1"]

    a_init = np.concatenate([
        np.array(session0["a_fast_init"], dtype=np.float32).reshape(-1),
        np.array(session1["a_fast_init"], dtype=np.float32).reshape(-1),
    ])
    state = bridge.reset(2, num_layers, dim, rank, a_init)
    for session_idx, trace in [(0, session0), (1, session1)]:
        for step in trace["steps"]:
            if step["op"] != "update":
                continue
            grad_a = np.array(step["grad_a"], dtype=np.float32)
            grad_b = np.array(step["grad_b"], dtype=np.float32)
            state = bridge.update(state, session_idx, step["layer"], grad_a, grad_b, lr=step["lr"], max_delta_norm=step["max_delta_norm"])

    decay0 = next(s["decay_rate"] for s in session0["steps"] if s["op"] == "decay")
    decay1 = next(s["decay_rate"] for s in session1["steps"] if s["op"] == "decay")
    decayed_for_0 = bridge.decay(state, decay0)
    decayed_for_1 = bridge.decay(state, decay1)
    a_fast = state.a_fast.copy()
    b_fast = state.b_fast.copy()
    a_fast[0] = decayed_for_0.a_fast[0]
    b_fast[0] = decayed_for_0.b_fast[0]
    a_fast[1] = decayed_for_1.a_fast[1]
    b_fast[1] = decayed_for_1.b_fast[1]
    state = bridge.FastWeightState(sessions=2, num_layers=num_layers, dim=dim, rank=rank, a_fast=a_fast, b_fast=b_fast, update_count=state.update_count)

    for session_idx, trace in [(0, session0), (1, session1)]:
        final = trace["final_state"]
        assert np.allclose(state.a_fast[session_idx], np.array(final["a_fast"], dtype=np.float32), atol=1e-4)
        assert np.allclose(state.b_fast[session_idx], np.array(final["b_fast"], dtype=np.float32), atol=1e-4)
        assert int(state.update_count[session_idx]) == final["update_count"]

        delta = bridge.effective_delta(state, session_idx, 0)
        assert np.allclose(delta, np.array(trace["final_delta_layer0"], dtype=np.float32), atol=1e-4)

        apply_data = trace["apply"]
        single = bridge.FastWeightState(
            sessions=1, num_layers=num_layers, dim=dim, rank=rank,
            a_fast=state.a_fast[session_idx:session_idx + 1], b_fast=state.b_fast[session_idx:session_idx + 1],
            update_count=state.update_count[session_idx:session_idx + 1],
        )
        y = bridge.apply(
            single, np.array(apply_data["x"], dtype=np.float32),
            np.array(apply_data["base_weight"], dtype=np.float32), np.array(apply_data["base_bias"], dtype=np.float32),
            apply_data["layer"],
        )
        assert np.allclose(y, np.array(apply_data["y"], dtype=np.float32), atol=1e-4)


def test_bridge_matches_live_python_reference_on_a_fresh_sequence():
    """Independent of the frozen fixture -- runs a NEW sequence through
    both the live Python/MLX reference and the bridge in the same test
    process, catching any drift between them a stale fixture could hide."""
    from reference.hz0d_fast_weights import (
        FastWeightConfig, apply_fast_linear, decay_fast_weights, effective_delta as py_effective_delta,
        init_fast_weights, update_fast_weights,
    )

    dim, rank, num_layers = 8, 2, 1
    config = FastWeightConfig(dim=dim, rank=rank, num_layers=num_layers, max_delta_norm=1.0, init_seed=7)
    py_state = init_fast_weights(config)

    bridge_state = bridge.reset(1, num_layers, dim, rank, np.array(py_state.a_fast).reshape(-1))
    assert np.allclose(bridge_state.a_fast[0], np.array(py_state.a_fast[0]), atol=1e-6)
    assert np.allclose(bridge_state.b_fast, np.zeros_like(bridge_state.b_fast))

    key = mx.random.key(42)
    for _ in range(3):
        k1, k2, key = mx.random.split(key, 3)
        grad_a = mx.random.normal((dim, rank), key=k1) * 0.4
        grad_b = mx.random.normal((rank, dim), key=k2) * 0.4
        py_state = update_fast_weights(py_state, 0, grad_a, grad_b, lr=0.05, config=config)
        bridge_state = bridge.update(bridge_state, 0, 0, np.array(grad_a).reshape(-1), np.array(grad_b).reshape(-1), lr=0.05, max_delta_norm=1.0)

    assert np.allclose(bridge_state.a_fast[0], np.array(py_state.a_fast[0]), atol=1e-4)
    assert np.allclose(bridge_state.b_fast[0], np.array(py_state.b_fast[0]), atol=1e-4)
    assert int(bridge_state.update_count[0]) == int(py_state.update_count)

    py_state = decay_fast_weights(py_state, 0.85)
    bridge_state = bridge.decay(bridge_state, 0.85)
    assert np.allclose(bridge_state.a_fast[0], np.array(py_state.a_fast[0]), atol=1e-4)
    assert np.allclose(bridge_state.b_fast[0], np.array(py_state.b_fast[0]), atol=1e-4)

    py_delta = py_effective_delta(py_state, 0)
    bridge_delta = bridge.effective_delta(bridge_state, 0, 0)
    assert np.allclose(bridge_delta, np.array(py_delta), atol=1e-4)

    key_x, key_w, key_b = mx.random.split(key, 3)
    x = mx.random.normal((1, dim), key=key_x) * 0.3
    base_weight = mx.random.normal((dim, dim), key=key_w) * 0.05
    base_bias = mx.random.normal((dim,), key=key_b) * 0.01
    py_y = apply_fast_linear(x, base_weight, base_bias, py_state, 0)
    bridge_y = bridge.apply(bridge_state, np.array(x), np.array(base_weight), np.array(base_bias), 0)
    assert np.allclose(bridge_y, np.array(py_y), atol=1e-4)


def test_reset_uses_the_same_a_fast_regardless_of_session_count():
    """A structural sanity check on the bridge's own batching contract:
    resetting 3 sessions with 3 independently-supplied `a_fast` blocks
    must give each session EXACTLY its own block back, not a mix."""
    dim, rank, num_layers = 4, 2, 1
    blocks = [np.full(num_layers * dim * rank, float(i + 1), dtype=np.float32) for i in range(3)]
    a_init = np.concatenate(blocks)
    state = bridge.reset(3, num_layers, dim, rank, a_init)
    for i in range(3):
        assert np.allclose(state.a_fast[i].reshape(-1), blocks[i])
