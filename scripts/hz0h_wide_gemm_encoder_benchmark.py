#!/usr/bin/env python3
"""Real GPU timing comparison for the wide-GEMM encoder layout remap
(reference/hz0h_bdh_wide_gemm_encoder_torch.py, Stage 1A of
plans/hatchlingzero_bdh_transformer_planning.md).

Isolates the ONE claim this remap makes: does reshaping BDH's per-head
encoder matmul into one big (B*T,D) x (D,H*N) GEMM run faster on real GPU
hardware than the oracle's own broadcasted per-head (nh separate (T,D)x
(D,N)) matmul, for the SAME weights and SAME inputs? This does not touch
training, the rest of the recurrent body, or claim an end-to-end BDH
speedup -- see the module docstring in the encoder extension file for the
disclosed scope limit (forward-only, not yet training-integrated).

Real, cheap-to-falsify by design: if the wide-GEMM layout doesn't measure
faster on the real hardware this project trains on, that's a genuine,
useful negative result -- report it, don't rationalize it away.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_wide_gemm_encoder_torch import (
    bdh_wide_gemm_encoder_step,
    wide_encoder_view,
)


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

    x = torch.randn(args.batch_size, 1, args.seq_len, args.n_embd, device=device, dtype=dtype)
    encoder_wide = wide_encoder_view(model.encoder.to(dtype))

    def broadcast_matmul():
        return x @ model._w(model.encoder)

    def wide_gemm():
        return bdh_wide_gemm_encoder_step(x, encoder_wide, args.n_head, N)

    with torch.no_grad():
        oracle_out = broadcast_matmul()
        wide_out = wide_gemm()
    max_diff = (oracle_out.float() - wide_out.float()).abs().max().item()

    with torch.no_grad():
        broadcast_seconds = time_forward(broadcast_matmul, args.warmup, args.steps, device)
        wide_seconds = time_forward(wide_gemm, args.warmup, args.steps, device)

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
        "wide_gemm_seconds": wide_seconds,
        "broadcast_matmul_steps_per_second": args.steps / broadcast_seconds,
        "wide_gemm_steps_per_second": args.steps / wide_seconds,
        "wide_gemm_speedup_ratio": broadcast_seconds / wide_seconds,
    }
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
