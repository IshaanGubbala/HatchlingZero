#!/usr/bin/env python3
"""Real GPU profiler cross-check for the analytic byte-ledger/roofline
result (scripts/hz0h_bdh_vb_subspace_decoder_byte_ledger.py, real
result: round is compute-bound, 3.18x-4.45x above the roofline knee at
both RTX 4090/5090, ASSUMING peak advertised TFLOPS is actually
achieved). This script measures whether that assumption holds on real
hardware -- if achieved GEMM throughput is much lower than the
advertised peak, the effective knee moves and the analytic conclusion
could flip.

Real, disclosed methodology (not nsight-level, deliberately scoped to
what torch.profiler gives cheaply):
- Achieved aggregate TFLOPS: measured wall-clock per step (CUDA-
  synchronized) divided into the analytic total FLOPs/step (forward +
  2x forward for a checkpointed backward's recompute, matching the
  byte ledger's own assumption) -- a real, direct throughput number,
  not a profiler-internal estimate.
- Achieved elementwise-op bandwidth: torch.profiler's per-kernel
  breakdown, filtered to kernel names matching relu/add/mul/
  layer_norm/copy (the low-arithmetic-intensity ops the ledger
  identified as carrying ~47% of round bytes for ~0% of FLOPs),
  summed CUDA self-time, divided into the ledger's analytic bytes for
  those same ops -- a real, targeted bandwidth measurement on exactly
  the portion of the workload expected to be memory-bound, not a
  blended/ambiguous whole-step number.
- GEMM vs elementwise CUDA time share, from the same per-kernel
  breakdown.
- torch.compile fusion check: kernel COUNT for the elementwise chain,
  eager vs compiled -- real fusion collapses several separate aten
  calls into fewer Triton kernels; if the count doesn't drop, Inductor
  isn't fusing this chain.
- Batch-size sensitivity: same measurement at B=1/2/8, checking for
  this project's own previously-documented B=1->B=2 kernel-dispatch
  cliff.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_vb_subspace_decoder_checkpointed_torch import bdh_vb_subspace_decoder_forward_checkpointed
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_vb_subspace_decoder_byte_ledger import build_ledger, GPU_SPECS

ELEMENTWISE_MARKERS = ["relu", "add", "mul", "layer_norm", "copy", "clamp", "fill", "native_layer_norm"]
GEMM_MARKERS = ["mm", "bmm", "gemm", "addmm", "baddbmm"]


def classify(name: str) -> str:
    lname = name.lower()
    if any(m in lname for m in GEMM_MARKERS):
        return "gemm"
    if any(m in lname for m in ELEMENTWISE_MARKERS):
        return "elementwise"
    return "other"


def run_profiled_step(model, idx, target, n_layer: int, device, compiled_fn, n_warmup: int, n_measure: int):
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        _, loss = compiled_fn(model, idx, n_layer, target)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(n_measure):
        _, loss = compiled_fn(model, idx, n_layer, target)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    wall_time = (time.perf_counter() - started) / n_measure

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        _, loss = compiled_fn(model, idx, n_layer, target)
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    events = prof.key_averages()
    kernel_stats = {"gemm": {"count": 0, "cuda_time_us": 0.0}, "elementwise": {"count": 0, "cuda_time_us": 0.0}, "other": {"count": 0, "cuda_time_us": 0.0}}
    top_kernels = []
    for e in events:
        cuda_time = getattr(e, "self_device_time_total", None)
        if cuda_time is None:
            cuda_time = getattr(e, "self_cuda_time_total", 0.0)
        if cuda_time <= 0:
            continue
        cls = classify(e.key)
        kernel_stats[cls]["count"] += e.count
        kernel_stats[cls]["cuda_time_us"] += cuda_time
        top_kernels.append((e.key, cuda_time, e.count))
    top_kernels.sort(key=lambda t: -t[1])

    return {
        "wall_time_s": wall_time,
        "kernel_stats": kernel_stats,
        "top_kernels": [{"name": n, "cuda_time_us": t, "count": c} for n, t, c in top_kernels[:15]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-measure", type=int, default=10)
    parser.add_argument("--test-compile", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "this diagnostic needs a real CUDA GPU"
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[profiler] real device: {gpu_name}", flush=True)

    config = BDHVBSubspaceDecoderConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                                         mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
                                         d_state=args.d_state, subspace_rank=args.subspace_rank)

    results = {}
    for B in args.batch_sizes:
        torch.manual_seed(7)
        model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.bfloat16)
        idx = torch.randint(0, 256, (B, args.sequence_length), device=device)
        target = torch.randint(0, 256, (B, args.sequence_length), device=device)

        ledger = build_ledger(B, args.sequence_length, args.n_embd, args.mult, args.n_head, args.d_state, args.subspace_rank)
        step_flops = ledger["totals"]["flops_per_round"] * args.n_layer * 3
        step_bytes = ledger["totals"]["total_bytes_per_round"] * args.n_layer * 2
        # Ledger op names are descriptive strings (e.g. "x_sparse = relu(x_latent)"),
        # not real profiler kernel names -- classify() is for the latter, matched
        # separately below. Here we just directly select the ledger's own
        # relu/LN/elementwise-multiply entries by their known text markers.
        ew_bytes_per_round = sum(o["total_bytes"] for o in ledger["ops"] if "LN" in o["op"] or "relu" in o["op"].lower() or "*" in o["op"])

        print(f"\n=== B={B} eager ===", flush=True)
        r = run_profiled_step(model, idx, target, args.n_layer, device, bdh_vb_subspace_decoder_forward_checkpointed,
                               args.n_warmup, args.n_measure)
        achieved_tflops = step_flops / r["wall_time_s"] / 1e12
        gemm_us = r["kernel_stats"]["gemm"]["cuda_time_us"]
        ew_us = r["kernel_stats"]["elementwise"]["cuda_time_us"]
        ew_bytes_total = ew_bytes_per_round * args.n_layer * 2
        achieved_ew_bw_gb_s = (ew_bytes_total / (ew_us / 1e6)) / 1e9 if ew_us > 0 else float("nan")
        print(f"  wall_time={r['wall_time_s']*1000:.2f}ms achieved={achieved_tflops:.1f} TFLOPS "
              f"gemm_us={gemm_us:.0f} elementwise_us={ew_us:.0f} elementwise_bw={achieved_ew_bw_gb_s:.1f} GB/s", flush=True)
        print(f"  top kernels: {[(k['name'], round(k['cuda_time_us'])) for k in r['top_kernels'][:8]]}", flush=True)

        entry = {"eager": {**r, "achieved_tflops": achieved_tflops, "achieved_elementwise_bw_gb_s": achieved_ew_bw_gb_s,
                            "step_flops": step_flops, "step_bytes": step_bytes}}

        if args.test_compile:
            print(f"=== B={B} torch.compile ===", flush=True)
            compiled_fn = torch.compile(bdh_vb_subspace_decoder_forward_checkpointed, mode="max-autotune")
            r2 = run_profiled_step(model, idx, target, args.n_layer, device, compiled_fn, args.n_warmup, args.n_measure)
            achieved_tflops2 = step_flops / r2["wall_time_s"] / 1e12
            print(f"  wall_time={r2['wall_time_s']*1000:.2f}ms achieved={achieved_tflops2:.1f} TFLOPS "
                  f"elementwise_kernel_count={r2['kernel_stats']['elementwise']['count']} (eager was {r['kernel_stats']['elementwise']['count']})", flush=True)
            entry["compiled"] = {**r2, "achieved_tflops": achieved_tflops2}

        results[f"B{B}"] = entry
        del model
        torch.cuda.empty_cache()

    for gpu, spec in GPU_SPECS.items():
        print(f"\n[roofline] {gpu} advertised peak={spec['peak_tflops_bf16']:.1f} TFLOPS, bandwidth={spec['bandwidth_gb_s']:.0f} GB/s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"gpu": gpu_name, "gpu_specs_reference": GPU_SPECS, "results": results}, indent=2, default=str), encoding="utf-8")
    print(f"\n[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
