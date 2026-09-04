#!/usr/bin/env python3
"""HZ-CQ-v1 real cross-platform speed benchmark, plan section 11.5's
"finish/benchmark the already-landed semantics-preserving packing/
caching on real MPS + CUDA, not just CPU" -- priority 1 of the
cross-platform speed queue. Compares the naive per-round path
(`HZCQReasoningWorkspace.step`, called manually in a loop -- no K/V
caching, no packed Q, items 1/5/6 all off) against the optimized path
(`run`, items 1/5/6 on) on whatever real accelerator this machine has.

Matches 11.5's benchmark contract: M_H=32 (the confirmed Pareto point,
not just M_H=8), R in {2,4,8,16}, batch in {1, a training-relevant
size}, forward-only AND full training-step (forward+backward+optim)
latency, throughput, peak memory, output/accuracy equivalence (bit-
identical check, same as the CPU verification already done for these
items). Device-aware: picks cuda > mps > cpu unless --device is given,
and labels every result row with the real device it ran on so a CPU
number can never be silently reported as this section's cross-platform
result (11.5's own explicit warning about the earlier 1.14x/1.125x
CPU-only numbers).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_hzcq_v1_reasoning_workspace_torch import HZCQReasoningWorkspace, HZCQReasoningWorkspaceConfig


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated()
    if device.type == "mps":
        return torch.mps.driver_allocated_memory()
    return None


def reset_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


def naive_run(ws: HZCQReasoningWorkspace, batch: int, S: torch.Tensor, x: torch.Tensor, n_rounds: int) -> torch.Tensor:
    """Items 1/5/6 all OFF: calls `step` (no K/V caching, no packed Q,
    S-summary recomputed every round) in a manual loop -- the pre-
    optimization baseline, kept alive unchanged in the reference module
    exactly so this comparison stays possible."""
    H = ws.init_state(batch, device=S.device, dtype=S.dtype)
    for _ in range(n_rounds):
        H = ws.step(H, S, x)
    return H


def optimized_run(ws: HZCQReasoningWorkspace, batch: int, S: torch.Tensor, x: torch.Tensor, n_rounds: int) -> torch.Tensor:
    """Items 1/5/6 all ON: the real `run()` path."""
    return ws.run(batch, S, x, n_rounds=n_rounds)


def time_forward(fn, ws, batch, S, x, n_rounds, device, warmup, reps):
    for _ in range(warmup):
        fn(ws, batch, S, x, n_rounds)
    sync(device)
    reset_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(ws, batch, S, x, n_rounds)
    sync(device)
    elapsed = time.perf_counter() - t0
    return elapsed / reps, out, peak_memory_bytes(device)


def time_train_step(fn, ws, opt, batch, S, x, n_rounds, device, warmup, reps):
    def step_once():
        opt.zero_grad(set_to_none=True)
        out = fn(ws, batch, S, x, n_rounds)
        loss = out.pow(2).mean()
        loss.backward()
        opt.step()
        return out

    for _ in range(warmup):
        step_once()
    sync(device)
    reset_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        step_once()
    sync(device)
    elapsed = time.perf_counter() - t0
    return elapsed / reps, peak_memory_bytes(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None, help="defaults to cuda > mps > cpu, whichever this machine has")
    parser.add_argument("--d-model", type=int, default=80)
    parser.add_argument("--workspace-slots", type=int, default=32, help="the confirmed M_H Pareto point, not just M_H=8")
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--r-values", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--query-len", type=int, default=20)
    parser.add_argument("--memory-len", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device_str = args.device or pick_device()
    device = torch.device(device_str)
    print(f"[bench] device={device}", flush=True)

    torch.manual_seed(args.seed)
    D = args.d_model
    ws = HZCQReasoningWorkspace(HZCQReasoningWorkspaceConfig(
        n_embd=D, workspace_slots=args.workspace_slots, gate_hidden=args.gate_hidden,
        allow_ablation_slots=args.workspace_slots > 8)).to(device)

    # Real equivalence check first -- do not trust any timing number
    # below until this passes (11.4's own required discipline for any
    # optimized path, applied here on the actual target device instead
    # of only CPU).
    g = torch.Generator().manual_seed(1)
    S_check = torch.randn(2, args.memory_len, D, generator=g).to(device)
    x_check = torch.randn(2, args.query_len, D, generator=g).to(device)
    with torch.no_grad():
        H_naive = naive_run(ws, 2, S_check, x_check, 12)
        H_opt = optimized_run(ws, 2, S_check, x_check, 12)
    bit_identical = torch.equal(H_naive, H_opt)
    max_diff = (H_naive - H_opt).abs().max().item()
    print(f"[bench] equivalence check on {device}: bit_identical={bit_identical} max_abs_diff={max_diff}", flush=True)
    if not bit_identical and max_diff > 1e-4:
        print(f"[bench] WARNING: naive/optimized paths disagree by more than float noise on {device} -- "
              f"do not trust the timing numbers below", flush=True)

    results = []
    for batch in args.batch_sizes:
        S = torch.randn(batch, args.memory_len, D, device=device)
        x = torch.randn(batch, args.query_len, D, device=device)
        for r in args.r_values:
            opt = torch.optim.AdamW(ws.parameters(), lr=1e-3)

            with torch.no_grad():
                naive_fwd_s, _, naive_fwd_mem = time_forward(naive_run, ws, batch, S, x, r, device, args.warmup, args.reps)
                opt_fwd_s, _, opt_fwd_mem = time_forward(optimized_run, ws, batch, S, x, r, device, args.warmup, args.reps)

            naive_train_s, naive_train_mem = time_train_step(naive_run, ws, opt, batch, S, x, r, device, args.warmup, args.reps)
            opt_train_s, opt_train_mem = time_train_step(optimized_run, ws, opt, batch, S, x, r, device, args.warmup, args.reps)

            row = {
                "device": str(device), "batch": batch, "r": r,
                "naive_forward_ms": naive_fwd_s * 1000, "optimized_forward_ms": opt_fwd_s * 1000,
                "forward_speedup": naive_fwd_s / opt_fwd_s if opt_fwd_s > 0 else None,
                "naive_train_step_ms": naive_train_s * 1000, "optimized_train_step_ms": opt_train_s * 1000,
                "train_step_speedup": naive_train_s / opt_train_s if opt_train_s > 0 else None,
                "naive_forward_throughput_eps": batch / naive_fwd_s,
                "optimized_forward_throughput_eps": batch / opt_fwd_s,
                "naive_train_peak_mem_bytes": naive_train_mem, "optimized_train_peak_mem_bytes": opt_train_mem,
            }
            results.append(row)
            print(f"[bench] batch={batch} R={r} fwd: naive={row['naive_forward_ms']:.3f}ms "
                  f"opt={row['optimized_forward_ms']:.3f}ms speedup={row['forward_speedup']:.3f}x | "
                  f"train_step: naive={row['naive_train_step_ms']:.3f}ms opt={row['optimized_train_step_ms']:.3f}ms "
                  f"speedup={row['train_step_speedup']:.3f}x", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "device": str(device),
        "equivalence_check": {"bit_identical": bit_identical, "max_abs_diff": max_diff},
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
