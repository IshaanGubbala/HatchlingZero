#!/usr/bin/env python3
"""Oracle-packed decoder ceiling benchmark -- the gate before building any
exact-sparse kernel.

Real x-density measured on our production checkpoint: 28.3% (results/
local/hz0h_bdh_x_sparsity_diagnostic.json). Since g_i=0 whenever x_s[i]=0
(an exact identity, not an approximation), the decoder's input columns
and the decoder's own rows corresponding to inactive x positions
contribute exactly zero to the output -- they can be skipped entirely.

This does NOT build the real per-token variable-length gather/routing
machinery (that's real engineering, and per this project's own closed
lanes, gather/scatter overhead has erased theoretical FLOP savings
before). Instead it answers the prior, cheaper question first: pre-gather
a fixed, density-matched subset of decoder rows/columns ONCE, OUTSIDE
the timed region (pretending routing is free), then time only the
resulting smaller dense GEMM against the full dense one. This is the
real ceiling -- if routing/indexing overhead could be made free, this is
the best case speedup available. Per the proposal: <1.2x means don't
build the kernel; 1.5-1.8x+ means it's justified.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARMS = ("dense", "oracle_packed")


def make_inputs(args, device):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    wide = args.n_head * args.n_per_head
    decoder_input = torch.randn(
        args.batch_size * args.sequence_length, wide, device=device, dtype=torch.bfloat16, generator=generator,
    )
    decoder = torch.randn(wide, args.n_embd, device=device, dtype=torch.bfloat16, generator=generator)
    n_active = max(1, int(round(wide * args.active_fraction)))
    active_idx = torch.randperm(wide, device=device, generator=generator)[:n_active].sort().values
    return decoder_input, decoder, active_idx


def run_child(args):
    device = torch.device("cuda")
    decoder_input, decoder, active_idx = make_inputs(args, device)

    if args.arm == "dense":
        def operation():
            return decoder_input @ decoder
    else:
        # Pre-gather ONCE, outside the timed region -- pretends routing/
        # index computation is free, isolating the smaller-GEMM speedup
        # from the real (separate, harder) cost of computing the gather
        # itself per token.
        packed_input = decoder_input.index_select(1, active_idx).contiguous()
        packed_decoder = decoder.index_select(0, active_idx).contiguous()

        def operation():
            return packed_input @ packed_decoder

    for _ in range(args.warmup):
        operation()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    seconds_per_step_trials = []
    output = None
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.steps):
            output = operation()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        seconds_per_step_trials.append(elapsed / args.steps)

    mean = sum(seconds_per_step_trials) / len(seconds_per_step_trials)
    variance = sum((t - mean) ** 2 for t in seconds_per_step_trials) / len(seconds_per_step_trials)
    result = {
        "arm": args.arm,
        "seconds_per_step_trials": seconds_per_step_trials,
        "seconds_per_step_mean": mean,
        "seconds_per_step_stdev": variance ** 0.5,
        "rows_per_second_mean": (args.batch_size * args.sequence_length) / mean,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "finite": bool(torch.isfinite(output).all()),
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


def child_command(args, arm, out):
    return [
        sys.executable, str(Path(__file__).resolve()),
        "--child", "--arm", arm, "--out", str(out),
        "--batch-size", str(args.batch_size),
        "--sequence-length", str(args.sequence_length),
        "--n-embd", str(args.n_embd),
        "--n-head", str(args.n_head),
        "--n-per-head", str(args.n_per_head),
        "--active-fraction", str(args.active_fraction),
        "--warmup", str(args.warmup),
        "--steps", str(args.steps),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
    ]


def run_parent(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    results = {}
    with tempfile.TemporaryDirectory(prefix="hz0h_oracle_packed_decoder_") as directory:
        for arm in ARMS:
            child_out = Path(directory) / f"{arm}.json"
            subprocess.run(child_command(args, arm, child_out), check=True)
            results[arm] = json.loads(child_out.read_text(encoding="utf-8"))
    dense = results["dense"]
    packed = results["oracle_packed"]
    report = {
        "experiment_id": "oracle_packed_decoder_ceiling_v1",
        "scope": "Decoder GEMM only (the biggest of the three x-skippable projections). "
                 "Pre-gathered active columns/rows OUTSIDE the timed region -- measures the "
                 "ceiling speedup if per-token routing overhead were free, not real end-to-end "
                 "speedup with real gather/scatter cost included.",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "shape": {
            "batch_size": args.batch_size, "sequence_length": args.sequence_length,
            "n_embd": args.n_embd, "n_head": args.n_head, "n_per_head": args.n_per_head,
            "decoder_width": args.n_head * args.n_per_head,
            "active_fraction": args.active_fraction,
        },
        "fresh_subprocess_per_arm": True,
        "arms": results,
        "packed_over_dense_speedup": packed["rows_per_second_mean"] / dense["rows_per_second_mean"],
        "packed_over_dense_allocated_memory": packed["peak_memory_allocated_bytes"] / dense["peak_memory_allocated_bytes"],
        "gate": "kernel engineering justified only if this speedup is >=1.5x; <1.2x means stop",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-per-head", type=int, default=4992)
    parser.add_argument("--active-fraction", type=float, default=0.283)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=list(ARMS), default="dense", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.child:
        run_child(parsed)
    else:
        run_parent(parsed)
