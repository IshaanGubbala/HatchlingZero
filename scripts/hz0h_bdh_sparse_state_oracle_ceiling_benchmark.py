#!/usr/bin/env python3
"""Go/no-go gate for Tier 3 items 19-20 (lazy RoPE, fused state kernel),
following the SAME oracle-packed methodology as item 11's decoder
benchmark (scripts/hz0h_bdh_oracle_packed_decoder_ceiling_benchmark.py):
pre-gather a density-matched active-row subset OUTSIDE the timed region
(pretending routing/indexing is free), time only the resulting smaller
GEMMs against the full dense state read+write.

Uses the REAL measured pair density from item 18 (47.0%, not the raw
28.6% coordinate density -- RoPE pairing is the real constraint here)
as the active fraction, at production shape. Per plan Constraint D:
"Every sparse/low-rank idea gets an oracle upper-bound benchmark before
custom kernel investment" -- this is that check for the state path
specifically, given item 17 already showed a real, severe (126x) loss
for a structurally similar gather-based approach.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, repeats: int, warmup: int, device: torch.device) -> dict:
    for _ in range(warmup):
        fn()
    _sync(device)
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync(device)
    elapsed = time.perf_counter() - started
    return {"seconds_per_call": elapsed / repeats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--active-fraction", type=float, default=0.470, help="Real measured post-RoPE pair density (item 18).")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    torch.manual_seed(args.seed)

    D = args.n_embd
    nh = args.n_head
    N = D * args.mult // nh
    B = args.batch_size

    state = torch.randn(B, nh, N, D, device=device, dtype=torch_dtype)
    QR = torch.randn(B, nh, 1, N, device=device, dtype=torch_dtype)
    KR = torch.randn(B, nh, 1, N, device=device, dtype=torch_dtype)
    V = torch.randn(B, 1, 1, D, device=device, dtype=torch_dtype)

    def dense_step():
        y = QR @ state  # (B, nh, 1, D) -- read
        return state + KR.mT @ V  # (B, nh, N, D) -- write (new state)

    n_active = max(1, int(round(N * args.active_fraction)))
    generator = torch.Generator(device=device).manual_seed(args.seed)
    active_idx = torch.randperm(N, device=device, generator=generator)[:n_active].sort().values

    state_packed = state.index_select(2, active_idx).contiguous()
    QR_packed = QR.index_select(3, active_idx).contiguous()
    KR_packed = KR.index_select(3, active_idx).contiguous()

    def oracle_packed_step():
        y = QR_packed @ state_packed  # (B, nh, 1, D) -- read, only active rows
        return state_packed + KR_packed.mT @ V  # (B, nh, n_active, D) -- write, only active rows

    dense_result = bench(dense_step, args.repeats, args.warmup, device)
    oracle_result = bench(oracle_packed_step, args.repeats, args.warmup, device)
    speedup = dense_result["seconds_per_call"] / oracle_result["seconds_per_call"]

    print(f"[bench] dense: {dense_result['seconds_per_call']*1e6:.2f} us/call", flush=True)
    print(f"[bench] oracle-packed ({args.active_fraction*100:.1f}% active): {oracle_result['seconds_per_call']*1e6:.2f} us/call", flush=True)
    print(f"[bench] ceiling speedup: {speedup:.3f}x", flush=True)

    report = {
        "config": {"n_embd": D, "n_head": nh, "mult": args.mult, "N_per_head": N, "batch_size": B,
                    "active_fraction": args.active_fraction, "dtype": args.dtype},
        "dense": dense_result, "oracle_packed": oracle_result, "ceiling_speedup": speedup,
        "gate": "kernel engineering justified only if this speedup is >=1.5x; <1.2x means stop (matches item 11's own gate)",
        "note": "pre-gathered indices OUTSIDE the timed region, pretending routing/indexing is free -- this is the "
                 "ceiling, not a real end-to-end measurement (item 17 already showed real gather overhead can turn "
                 "a good ceiling into a 126x real-world loss).",
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
