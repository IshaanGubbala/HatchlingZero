"""HZ-0B B10: honest wall-clock benchmark, Python/MLX reference vs the
Rust ctypes bridge, at the scale B6-B9's probes have actually used
(num_slots=16, key_dim=value_dim=32, batch=1 -- see
`docs/restart/hz0b_b6_real_integration_results.md` etc.) -- used to make
a real go/no-go call on building Metal GPU kernels (design doc's step 2)
instead of assuming either way.

Each iteration does one write + one read + one forget_or_decay, the same
three ops any real training step exercises per position.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "restart" / "hz0a_pmetal" / "python"))

NUM_SLOTS, KEY_DIM, VALUE_DIM = 16, 32, 32
ITERS = 2000


def bench_python_mlx() -> float:
    import mlx.core as mx

    from reference.hz0b_memory_simulator import forget_or_decay, read, reset, write

    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    for i in range(ITERS):
        key = mx.array(rng.standard_normal((1, KEY_DIM)).astype(np.float32))
        value = mx.array(rng.standard_normal((1, VALUE_DIM)).astype(np.float32))
        state, slot, _ = write(state, key, value, mx.array([0.9]), step=i, source=1)
        _readout, _weights = read(state, key, hard=True)
        state = forget_or_decay(state, decay_rate=0.98)
        mx.eval(state.keys, state.values, state.confidence)
    return time.perf_counter() - start


def bench_rust_bridge() -> float:
    import hz0b_memory_bridge as bridge

    state = bridge.reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    for i in range(ITERS):
        key = rng.standard_normal((1, KEY_DIM)).astype(np.float32)
        value = rng.standard_normal((1, VALUE_DIM)).astype(np.float32)
        state, slot, _ = bridge.write(state, key, value, np.array([0.9], dtype=np.float32), step=i, source=1)
        _readout, _weights = bridge.read(state, key, hard=True)
        state = bridge.forget_or_decay(state, decay_rate=0.98)
    return time.perf_counter() - start


def main():
    print(f"scale: num_slots={NUM_SLOTS} key_dim={KEY_DIM} value_dim={VALUE_DIM} batch=1, {ITERS} iterations (write+read+decay each)")
    mlx_time = bench_python_mlx()
    print(f"Python/MLX reference: {mlx_time:.4f}s total, {1000 * mlx_time / ITERS:.4f}ms/iteration")
    rust_time = bench_rust_bridge()
    print(f"Rust bridge (ctypes): {rust_time:.4f}s total, {1000 * rust_time / ITERS:.4f}ms/iteration")
    print(f"speedup: {mlx_time / rust_time:.2f}x")


if __name__ == "__main__":
    main()
