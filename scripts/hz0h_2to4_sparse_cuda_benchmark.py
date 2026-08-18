#!/usr/bin/env python3
"""Real CUDA training-step benchmark for 2:4 structured-sparsity BDH,
with real energy (joules/token) tracking wired in.

Real, disclosed scope: compares three arms --
  raw:            unmodified dense BDH.forward
  pruned_dense:   2:4-pruned weights (real quality-affecting change),
                  executed as plain dense matmuls (no hardware sparse
                  GEMM) -- isolates the QUALITY cost of pruning from any
                  speed effect, since dense-executing a pruned matrix
                  should be numerically identical in cost to raw.
  pruned_sparse:  same 2:4-pruned weights, but encoder+decoder execute
                  through the real hardware sparse Tensor Core path
                  (bdh_2to4_semi_structured_forward) -- encoder_v stays
                  dense (real, disclosed limitation, see that function's
                  own docstring). This is the only arm that can show a
                  real hardware speed win; the other two exist to
                  separate "did pruning change quality" from "did the
                  sparse kernel change speed."

This Mac has no CUDA -- this script cannot run here. Syntax-checked and
dispatched for a real Windows/RTX3060 run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from reference.hz0h_bdh_2to4_sparse_torch import (
    apply_2to4_pruning_to_bdh,
    bdh_2to4_semi_structured_forward,
)
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_energy import TrainingEnergySampler


def _run_arm(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    config = BDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
        vocab_size=256, dropout=0.0,
    )
    model = BDH(config).to(device=device, dtype=torch.bfloat16)
    model.attn.freqs = model.attn.freqs.to(torch.float32)

    if args.arm in ("pruned_dense", "pruned_sparse"):
        model = apply_2to4_pruning_to_bdh(model)

    idx = torch.randint(256, (args.batch_size, args.sequence_length), device=device)
    targets = torch.randint(256, idx.shape, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1, fused=True)

    def forward(token_ids, target_ids):
        if args.arm == "pruned_sparse":
            return bdh_2to4_semi_structured_forward(model, token_ids, target_ids)
        return model(token_ids, target_ids)

    def step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        _logits, loss = forward(idx, targets)
        loss.backward()
        optimizer.step()
        return loss.detach()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    sampler = TrainingEnergySampler(interval_seconds=0.1)
    sampler.start()
    started = time.perf_counter()
    losses = [float(step()) for _ in range(args.steps)]
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    tokens = idx.numel() * args.steps
    energy_report = sampler.stop(tokens=tokens)
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    return {
        "arm": args.arm,
        "tokens_per_second": tokens / seconds,
        "milliseconds_per_step": seconds * 1000.0 / args.steps,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "last_loss": losses[-1],
        "finite": bool(torch.isfinite(torch.tensor(losses)).all()) and gradients_finite,
        **energy_report,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run-arm", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=("raw", "pruned_dense", "pruned_sparse"), default="raw")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.run_arm:
        print(json.dumps(_run_arm(args)))
        return
    if args.out is None:
        raise ValueError("--out is required")

    arms = {}
    stderr = {}
    for arm in ("raw", "pruned_dense", "pruned_sparse"):
        command = [
            sys.executable, str(Path(__file__).resolve()), "--run-arm", "--arm", arm,
            "--batch-size", str(args.batch_size), "--sequence-length", str(args.sequence_length),
            "--n-embd", str(args.n_embd), "--n-layer", str(args.n_layer), "--n-head", str(args.n_head),
            "--mlp-internal-dim-multiplier", str(args.mlp_internal_dim_multiplier),
            "--warmup", str(args.warmup), "--steps", str(args.steps), "--seed", str(args.seed),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.stderr.strip():
            stderr[arm] = completed.stderr.strip()
        if completed.returncode != 0:
            arms[arm] = {"arm": arm, "error": "nonzero exit", "returncode": completed.returncode}
            continue
        arms[arm] = json.loads(completed.stdout.strip().splitlines()[-1])

    report = {
        "device": "cuda",
        "dtype": "bfloat16",
        "fresh_subprocess_per_arm": True,
        "shape": {
            "batch_size": args.batch_size, "sequence_length": args.sequence_length,
            "n_embd": args.n_embd, "n_layer": args.n_layer, "n_head": args.n_head,
            "mlp_internal_dim_multiplier": args.mlp_internal_dim_multiplier,
        },
        "warmup": args.warmup,
        "steps": args.steps,
        "arms": arms,
        "child_stderr": stderr,
    }
    if "error" not in arms.get("pruned_sparse", {}) and "error" not in arms.get("raw", {}):
        report["pruned_sparse_over_raw_throughput_ratio"] = (
            arms["pruned_sparse"]["tokens_per_second"] / arms["raw"]["tokens_per_second"]
        )
        report["pruned_sparse_over_raw_peak_memory_ratio"] = (
            arms["pruned_sparse"]["peak_memory_bytes"] / arms["raw"]["peak_memory_bytes"]
        )
    if "error" not in arms.get("pruned_dense", {}) and "error" not in arms.get("raw", {}):
        report["pruned_dense_over_raw_throughput_ratio"] = (
            arms["pruned_dense"]["tokens_per_second"] / arms["raw"]["tokens_per_second"]
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
