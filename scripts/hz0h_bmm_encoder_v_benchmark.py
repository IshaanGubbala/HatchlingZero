#!/usr/bin/env python3
"""Real GPU timing comparison for the explicit-bmm encoder_v layout remap
(reference/hz0h_bdh_bmm_encoder_v_torch.py, Stage 1A/1B of
plans/hatchlingzero_bdh_transformer_planning.md).

Isolates the one claim this remap makes: does an explicit per-head batched
GEMM (torch.bmm, head as the batch dim) run faster than BDH's own
broadcasted per-head matmul for the SAME weights and SAME inputs? Unlike
the encoder remap, this canNOT collapse into a single GEMM (each head's
input is genuinely different), so the honest open question here is
narrower: does cuBLAS already optimize the broadcast expression as well as
an explicit bmm, or is there a real gap? Report whichever the real numbers
show.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_bmm_encoder_v_torch import bmm_encoder_v_step
from reference.hz0h_bdh_torch import BDH, BDHConfig


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_forward(fn, warmup: int, steps: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    started = time.perf_counter()
    for _ in range(steps):
        fn()
    sync(device)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    torch.manual_seed(args.seed)

    config = BDHConfig(
        n_embd=args.n_embd, n_head=args.n_head, mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
    )
    model = BDH(config).to(device=device, dtype=dtype)
    N = args.n_embd * args.mlp_internal_dim_multiplier // args.n_head

    yKV = torch.randn(args.batch_size, args.n_head, args.seq_len, args.n_embd, device=device, dtype=dtype)

    def broadcast_matmul():
        return yKV @ model._w(model.encoder_v)

    def bmm_matmul():
        return bmm_encoder_v_step(yKV, model.encoder_v)

    with torch.no_grad():
        oracle_out = broadcast_matmul()
        bmm_out = bmm_matmul()
    max_diff = (oracle_out.float() - bmm_out.float()).abs().max().item()

    with torch.no_grad():
        broadcast_seconds = time_forward(broadcast_matmul, args.warmup, args.steps, device)
        bmm_seconds = time_forward(bmm_matmul, args.warmup, args.steps, device)

    report = {
        "device": str(device),
        "dtype": args.dtype,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "mlp_internal_dim_multiplier": args.mlp_internal_dim_multiplier,
        "N_per_head": N,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "warmup": args.warmup,
        "steps": args.steps,
        "parity_max_abs_diff": max_diff,
        "broadcast_matmul_seconds": broadcast_seconds,
        "bmm_seconds": bmm_seconds,
        "broadcast_matmul_steps_per_second": args.steps / broadcast_seconds,
        "bmm_steps_per_second": args.steps / bmm_seconds,
        "bmm_speedup_ratio": broadcast_seconds / bmm_seconds,
    }
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
