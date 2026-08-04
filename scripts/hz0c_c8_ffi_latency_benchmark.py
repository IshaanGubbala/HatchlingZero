"""HZ-0C C8: does the real Python<->Rust FFI bridge
(`reference/hz0c_pmetal_bridge.py`) actually run faster than the MLX
reference at production scale? Measured honestly, not assumed.

`docs/restart/hz0c_c9_end_to_end_report_results.md` found the MLX
reference path shows NO latency benefit from trigger sparsity (it's an
additively-masked full O(seq^2) computation). The FFI bridge now exists
and is proven correct (`tests/reference/test_hz0c_pmetal_bridge.py`), so
this measures whether routing through it, at the model's real dim=768/
heads=12 shape, is actually faster -- and reports the answer whichever
way it goes.

Two backends are benchmarked:
- CPU (`conditional_attention_forward`): the UNOPTIMIZED, correctness-
  first scalar Rust loop (no SIMD/BLAS) -- first measured here
  2026-08-04, found 35-190x SLOWER than MLX.
- GPU (`MetalConditionalAttention`, real Metal dispatch via a reusable
  handle, added the same day as the named next step after that CPU
  result): whether a genuine on-device GPU kernel closes that gap is an
  open, real question this benchmark answers rather than assumes.
"""
from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import numpy as np

from reference.hz0c_pmetal_bridge import MetalConditionalAttention, conditional_attention_forward
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


def time_gpu_ffi(gpu: MetalConditionalAttention, inputs: dict, trigger: np.ndarray, heads: int, *, repeats: int, warmup: int) -> dict:
    # The handle is created ONCE by the caller and reused across every
    # timed call here -- Metal device/pipeline/queue setup is real,
    # amortizable cost that must not be smuggled into a per-call number,
    # matching how MLX itself keeps its compiled kernels resident.
    for _ in range(warmup):
        gpu.forward(inputs["x"], trigger, qkv_w=inputs["qkv_w"], qkv_b=inputs["qkv_b"], out_w=inputs["out_w"], out_b=inputs["out_b"], heads=heads)
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        gpu.forward(inputs["x"], trigger, qkv_w=inputs["qkv_w"], qkv_b=inputs["qkv_b"], out_w=inputs["out_w"], out_b=inputs["out_b"], heads=heads)
        timings.append(time.perf_counter() - started)
    timings = np.asarray(timings)
    return {"mean_seconds": float(timings.mean()), "std_seconds": float(timings.std())}


def main(repeats: int = 10, warmup: int = 3, include_gpu: bool = True) -> None:
    shapes = [
        {"name": "c3_scenario_scale", "batch": 1, "seq": 40, "dim": 768, "heads": 12},
        {"name": "production_128", "batch": 1, "seq": 128, "dim": 768, "heads": 12},
    ]
    rates = [0.0, 0.15, 1.0]
    report = {"repeats": repeats, "warmup": warmup, "results": []}
    gpu = MetalConditionalAttention() if include_gpu else None
    try:
        for shape in shapes:
            inputs = make_inputs(shape["batch"], shape["seq"], shape["dim"], seed=555)
            for rate in rates:
                trigger = trigger_at_rate(shape["batch"], shape["seq"], rate)
                mlx_timing = time_mlx(inputs, trigger, shape["heads"], repeats=repeats, warmup=warmup)
                ffi_timing = time_ffi(inputs, trigger, shape["heads"], repeats=repeats, warmup=warmup)
                entry = {
                    "shape": shape["name"], "seq": shape["seq"], "dim": shape["dim"], "rate": rate,
                    "mlx_mean_seconds": mlx_timing["mean_seconds"], "mlx_std_seconds": mlx_timing["std_seconds"],
                    "cpu_ffi_mean_seconds": ffi_timing["mean_seconds"], "cpu_ffi_std_seconds": ffi_timing["std_seconds"],
                    "cpu_ffi_speedup_vs_mlx": mlx_timing["mean_seconds"] / ffi_timing["mean_seconds"],
                    "cpu_ffi_faster": ffi_timing["mean_seconds"] < mlx_timing["mean_seconds"],
                }
                if gpu is not None:
                    gpu_timing = time_gpu_ffi(gpu, inputs, trigger, shape["heads"], repeats=repeats, warmup=warmup)
                    entry["gpu_ffi_mean_seconds"] = gpu_timing["mean_seconds"]
                    entry["gpu_ffi_std_seconds"] = gpu_timing["std_seconds"]
                    entry["gpu_ffi_speedup_vs_mlx"] = mlx_timing["mean_seconds"] / gpu_timing["mean_seconds"]
                    entry["gpu_ffi_faster"] = gpu_timing["mean_seconds"] < mlx_timing["mean_seconds"]
                report["results"].append(entry)
    finally:
        if gpu is not None:
            gpu.close()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--no-gpu", action="store_true", help="skip the GPU FFI backend (e.g. no Metal device available)")
    args = parser.parse_args()
    main(args.repeats, args.warmup, include_gpu=not args.no_gpu)
