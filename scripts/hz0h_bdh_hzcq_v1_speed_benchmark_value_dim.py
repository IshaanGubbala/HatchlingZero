#!/usr/bin/env python3
"""HZ-CQ-v1 real cross-platform speed benchmark: full-D value/write
path (baseline) vs the SPEED-D half-width ablation (value_dim=D/2).
Plan section 11.5, SPEED-D item. Builds on top of the already-landed
K/V caching (item 1), S-summary caching (item 5), and packed-Q (item
6) -- both arms of this comparison run through the real `run()` path,
never the naive per-round `step()` loop.

Explicit device verification, per the project's own recent real
finding (device-placement bugs silently ran GPU dispatches on CPU):
prints device, GPU name (when CUDA), dtype, batch, and R for every
row -- a CUDA speedup number from this script cannot silently be a
CPU number, because the device is printed and stored with every
measurement.
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


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{torch.cuda.get_device_name(device)}"
    if device.type == "mps":
        return "mps:Apple Silicon GPU"
    return "cpu"


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_memory_bytes(device: torch.device):
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


def projection_flops(D: int, value_dim: int, M_H: int, M_S: int, T_query: int) -> int:
    """Real, estimated FLOPs for the V-projection + write_proj GEMMs
    only (the two SPEED-D actually changes) -- 2*in*out per Linear
    (multiply-add), summed over one round's read_s + read_x + write_proj.
    Not the whole model's FLOPs, just the piece this ablation targets."""
    v_s = 2 * D * value_dim * M_S
    v_x = 2 * D * value_dim * T_query
    write = 2 * (2 * value_dim) * D * M_H
    return v_s + v_x + write


def time_forward(ws, batch, S, x, n_rounds, device, warmup, reps):
    for _ in range(warmup):
        ws.run(batch, S, x, n_rounds=n_rounds)
    sync(device)
    reset_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        ws.run(batch, S, x, n_rounds=n_rounds)
    sync(device)
    elapsed = (time.perf_counter() - t0) / reps
    return elapsed, peak_memory_bytes(device)


def time_forward_backward(ws, batch, S, x, n_rounds, device, warmup, reps):
    def fwd_bwd():
        H = ws.run(batch, S, x, n_rounds=n_rounds)
        loss = H.pow(2).mean()
        loss.backward()
        for p in ws.parameters():
            p.grad = None

    for _ in range(warmup):
        fwd_bwd()
    sync(device)
    reset_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fwd_bwd()
    sync(device)
    elapsed = (time.perf_counter() - t0) / reps
    return elapsed, peak_memory_bytes(device)


def time_train_step(ws, opt, batch, S, x, n_rounds, device, warmup, reps):
    def step_once():
        opt.zero_grad(set_to_none=True)
        H = ws.run(batch, S, x, n_rounds=n_rounds)
        loss = H.pow(2).mean()
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step_once()
    sync(device)
    reset_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        step_once()
    sync(device)
    elapsed = (time.perf_counter() - t0) / reps
    return elapsed, peak_memory_bytes(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None, help="defaults to cuda > mps > cpu")
    parser.add_argument("--d-model", type=int, default=80)
    parser.add_argument("--workspace-slots", type=int, default=32, help="confirmed M_H Pareto point")
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--memory-len", type=int, default=8)
    parser.add_argument("--query-len", type=int, default=20)
    parser.add_argument("--r-values", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device or pick_device())
    dtype = getattr(torch, args.dtype)
    label = device_label(device)
    print(f"[bench] device={device} label={label} dtype={dtype}", flush=True)

    D = args.d_model
    value_dim = D // 2

    torch.manual_seed(args.seed)
    ws_baseline = HZCQReasoningWorkspace(HZCQReasoningWorkspaceConfig(
        n_embd=D, workspace_slots=args.workspace_slots, gate_hidden=args.gate_hidden,
        allow_ablation_slots=args.workspace_slots > 8)).to(device=device, dtype=dtype)
    torch.manual_seed(args.seed)
    ws_half = HZCQReasoningWorkspace(HZCQReasoningWorkspaceConfig(
        n_embd=D, workspace_slots=args.workspace_slots, gate_hidden=args.gate_hidden,
        allow_ablation_slots=args.workspace_slots > 8, value_dim=value_dim)).to(device=device, dtype=dtype)

    n_params_baseline = sum(p.numel() for p in ws_baseline.parameters())
    n_params_half = sum(p.numel() for p in ws_half.parameters())
    print(f"[bench] params baseline={n_params_baseline} half={n_params_half} "
          f"reduction={(1 - n_params_half/n_params_baseline)*100:.1f}%", flush=True)

    results = []
    for batch in args.batch_sizes:
        S = torch.randn(batch, args.memory_len, D, device=device, dtype=dtype)
        x = torch.randn(batch, args.query_len, D, device=device, dtype=dtype)
        for r in args.r_values:
            row = {"device": str(device), "device_label": label, "dtype": args.dtype, "batch": batch, "r": r}

            for name, ws in (("baseline", ws_baseline), ("half", ws_half)):
                with torch.no_grad():
                    fwd_s, fwd_mem = time_forward(ws, batch, S, x, r, device, args.warmup, args.reps)
                fwdbwd_s, fwdbwd_mem = time_forward_backward(ws, batch, S, x, r, device, args.warmup, args.reps)
                opt = torch.optim.AdamW(ws.parameters(), lr=1e-3)
                train_s, train_mem = time_train_step(ws, opt, batch, S, x, r, device, args.warmup, args.reps)
                row[f"{name}_forward_ms"] = fwd_s * 1000
                row[f"{name}_forward_backward_ms"] = fwdbwd_s * 1000
                row[f"{name}_train_step_ms"] = train_s * 1000
                row[f"{name}_forward_throughput_eps"] = batch / fwd_s
                row[f"{name}_train_peak_mem_bytes"] = train_mem

            row["forward_speedup"] = row["baseline_forward_ms"] / row["half_forward_ms"]
            row["forward_backward_speedup"] = row["baseline_forward_backward_ms"] / row["half_forward_backward_ms"]
            row["train_step_speedup"] = row["baseline_train_step_ms"] / row["half_train_step_ms"]
            row["v_write_flops_baseline"] = projection_flops(D, D, args.workspace_slots, args.memory_len, args.query_len)
            row["v_write_flops_half"] = projection_flops(D, value_dim, args.workspace_slots, args.memory_len, args.query_len)
            results.append(row)
            print(f"[bench] batch={batch:2d} R={r:2d} fwd_speedup={row['forward_speedup']:.3f}x "
                  f"fwd_bwd_speedup={row['forward_backward_speedup']:.3f}x "
                  f"train_speedup={row['train_step_speedup']:.3f}x", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "device": str(device), "device_label": label,
        "n_params_baseline": n_params_baseline, "n_params_half": n_params_half,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
