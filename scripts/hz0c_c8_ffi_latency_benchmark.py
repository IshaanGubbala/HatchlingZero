"""HZ-0C C8: does the real Python<->Rust FFI bridge
(`reference/hz0c_pmetal_bridge.py`) actually run faster than the MLX
reference at production scale? Measured honestly, not assumed.

`docs/restart/hz0c_c9_end_to_end_report_results.md` found the MLX
reference path shows NO latency benefit from trigger sparsity (it's an
additively-masked full O(seq^2) computation). The FFI bridge now exists
and is proven correct (`tests/reference/test_hz0c_pmetal_bridge.py`), so
this measures whether routing through it, at the model's real dim=768/
heads=12 shape, is actually faster -- and reports the answer whichever
way it goes. The underlying Rust kernel
(`hz0a_pmetal_kernel::conditional_anchor_attention_f32`) is an
UNOPTIMIZED, correctness-first, scalar CPU loop (no SIMD/BLAS) -- this
benchmark is exactly what would reveal if that matters in practice before
any claim of a production speedup is made.
"""
from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import numpy as np

from reference.hz0c_pmetal_bridge import conditional_attention_forward
from reference.hz0c_surprise_trigger import masked_anchor_attention


def make_inputs(batch: int, seq: int, dim: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, seq, dim)).astype(np.float32) * 0.3
    qkv_w = rng.normal(size=(3 * dim, dim)).astype(np.float32) * 0.02
    qkv_b = rng.normal(size=(3 * dim,)).astype(np.float32) * 0.02
    out_w = rng.normal(size=(dim, dim)).astype(np.float32) * 0.02
    out_b = rng.normal(size=(dim,)).astype(np.float32) * 0.02
    return {"x": x, "qkv_w": qkv_w, "qkv_b": qkv_b, "out_w": out_w, "out_b": out_b}


def trigger_at_rate(batch: int, seq: int, rate: float) -> np.ndarray:
    count = max(1, round(rate * seq))
    row = np.zeros(seq, dtype=np.float32)
    row[:count] = 1.0
    return np.broadcast_to(row, (batch, seq)).copy()


def time_mlx(inputs: dict, trigger: np.ndarray, heads: int, *, repeats: int, warmup: int) -> dict:
    x = mx.array(inputs["x"])
    qkv_w = mx.array(inputs["qkv_w"])
    qkv_b = mx.array(inputs["qkv_b"])
    out_w = mx.array(inputs["out_w"])
    out_b = mx.array(inputs["out_b"])
    trigger_mx = mx.array(trigger)
    for _ in range(warmup):
        out = masked_anchor_attention(x, trigger_mx, qkv_w=qkv_w, qkv_b=qkv_b, out_w=out_w, out_b=out_b, heads=heads)
        mx.eval(out)
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        out = masked_anchor_attention(x, trigger_mx, qkv_w=qkv_w, qkv_b=qkv_b, out_w=out_w, out_b=out_b, heads=heads)
        mx.eval(out)
        timings.append(time.perf_counter() - started)
    timings = np.asarray(timings)
    return {"mean_seconds": float(timings.mean()), "std_seconds": float(timings.std())}


def time_ffi(inputs: dict, trigger: np.ndarray, heads: int, *, repeats: int, warmup: int) -> dict:
    for _ in range(warmup):
        conditional_attention_forward(inputs["x"], trigger, qkv_w=inputs["qkv_w"], qkv_b=inputs["qkv_b"], out_w=inputs["out_w"], out_b=inputs["out_b"], heads=heads)
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        conditional_attention_forward(inputs["x"], trigger, qkv_w=inputs["qkv_w"], qkv_b=inputs["qkv_b"], out_w=inputs["out_w"], out_b=inputs["out_b"], heads=heads)
        timings.append(time.perf_counter() - started)
    timings = np.asarray(timings)
    return {"mean_seconds": float(timings.mean()), "std_seconds": float(timings.std())}


def main(repeats: int = 10, warmup: int = 3) -> None:
    shapes = [
        {"name": "c3_scenario_scale", "batch": 1, "seq": 40, "dim": 768, "heads": 12},
        {"name": "production_128", "batch": 1, "seq": 128, "dim": 768, "heads": 12},
    ]
    rates = [0.0, 0.15, 1.0]
    report = {"repeats": repeats, "warmup": warmup, "results": []}
    for shape in shapes:
        inputs = make_inputs(shape["batch"], shape["seq"], shape["dim"], seed=555)
        for rate in rates:
            trigger = trigger_at_rate(shape["batch"], shape["seq"], rate)
            mlx_timing = time_mlx(inputs, trigger, shape["heads"], repeats=repeats, warmup=warmup)
            ffi_timing = time_ffi(inputs, trigger, shape["heads"], repeats=repeats, warmup=warmup)
            speedup = mlx_timing["mean_seconds"] / ffi_timing["mean_seconds"]
            report["results"].append({
                "shape": shape["name"], "seq": shape["seq"], "dim": shape["dim"], "rate": rate,
                "mlx_mean_seconds": mlx_timing["mean_seconds"], "mlx_std_seconds": mlx_timing["std_seconds"],
                "ffi_mean_seconds": ffi_timing["mean_seconds"], "ffi_std_seconds": ffi_timing["std_seconds"],
                "ffi_speedup_vs_mlx": speedup,
                "ffi_faster": speedup > 1.0,
            })
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    main(args.repeats, args.warmup)
