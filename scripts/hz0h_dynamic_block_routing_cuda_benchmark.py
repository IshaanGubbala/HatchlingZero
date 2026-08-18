#!/usr/bin/env python3
"""Real CUDA training-step benchmark for per-token dynamic block routing
(reference/hz0h_bdh_dynamic_block_routing_layer_torch.py), sweeping real
capacity_factor values to see how speed/memory scale with the real,
data-dependent drop rate -- mirroring PackedBlockBDH's own real
active_fraction sweep methodology.

Real, disclosed scope: only the encoder projection is dynamically
routed; encoder_v/decoder stay dense (see that file's own docstring).
This measures the real cost/benefit of THIS specific, partial mechanism
-- not a full dynamically-routed BDH.

Correctness (the real exactness/gradient tests) already passed on CPU
this session; this script assumes that gate and focuses purely on real
CUDA speed/memory/finite-gradient measurement, one arm per fresh
subprocess (the isolation this session already proved necessary to
avoid cuBLAS co-residency artifacts).
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

from reference.hz0h_bdh_dynamic_block_routing_layer_torch import (
    dynamic_block_routing_forward,
    init_dynamic_block_router,
)
from reference.hz0h_bdh_torch import BDH, BDHConfig


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

    idx = torch.randint(256, (args.batch_size, args.sequence_length), device=device)
    targets = torch.randint(256, idx.shape, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1, fused=True)

    n_blocks = (args.n_embd * args.mlp_internal_dim_multiplier // args.n_head) // args.block_size
    router = None
    if args.arm == "dynamic_routing":
        router = init_dynamic_block_router(
            args.n_head, args.n_embd, n_blocks, generator=torch.Generator().manual_seed(args.seed),
        ).to(device=device, dtype=torch.bfloat16)
        router.requires_grad_(True)
        optimizer.add_param_group({"params": [router]})

    real_drop_rates = []

    def step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        if args.arm == "raw":
            _, loss = model(idx, targets)
        else:
            _, loss, routing_per_layer = dynamic_block_routing_forward(
                model, idx, router, targets, block_size=args.block_size,
                top_k=args.top_k, capacity_factor=args.capacity_factor,
            )
            total_picks = args.batch_size * args.sequence_length * args.top_k * args.n_head * args.n_layer
            total_dropped = sum(r.tokens_dropped for layer in routing_per_layer for r in layer)
            real_drop_rates.append(total_dropped / total_picks if total_picks else 0.0)
        loss.backward()
        optimizer.step()
        return loss.detach()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = [float(step()) for _ in range(args.steps)]
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    tokens = idx.numel() * args.steps
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    if router is not None:
        gradients_finite = gradients_finite and bool(torch.isfinite(router.grad).all())
    return {
        "arm": args.arm,
        "capacity_factor": args.capacity_factor if args.arm != "raw" else None,
        "tokens_per_second": tokens / seconds,
        "milliseconds_per_step": seconds * 1000.0 / args.steps,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "last_loss": losses[-1],
        "finite": bool(torch.isfinite(torch.tensor(losses)).all()) and gradients_finite,
        "mean_real_drop_rate": sum(real_drop_rates) / len(real_drop_rates) if real_drop_rates else None,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run-arm", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=("raw", "dynamic_routing"), default="raw")
    parser.add_argument("--capacity-factor", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.run_arm:
        print(json.dumps(_run_arm(args)))
        return
    if args.out is None:
        raise ValueError("--out is required")

    common = [
        "--batch-size", str(args.batch_size), "--sequence-length", str(args.sequence_length),
        "--n-embd", str(args.n_embd), "--n-layer", str(args.n_layer), "--n-head", str(args.n_head),
        "--mlp-internal-dim-multiplier", str(args.mlp_internal_dim_multiplier),
        "--block-size", str(args.block_size), "--top-k", str(args.top_k),
        "--warmup", str(args.warmup), "--steps", str(args.steps), "--seed", str(args.seed),
    ]

    def run_arm(arm: str, capacity_factor: float | None = None) -> dict:
        command = [sys.executable, str(Path(__file__).resolve()), "--run-arm", "--arm", arm] + common
        if capacity_factor is not None:
            command += ["--capacity-factor", str(capacity_factor)]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            return {"arm": arm, "capacity_factor": capacity_factor, "error": completed.stderr.strip()[-4000:]}
        return json.loads(completed.stdout.strip().splitlines()[-1])

    arms = {"raw": run_arm("raw")}
    for capacity_factor in (2.0, 1.0, 0.5, 0.25):
        arms[f"dynamic_routing_cf_{capacity_factor}"] = run_arm("dynamic_routing", capacity_factor)

    report = {
        "device": "cuda",
        "dtype": "bfloat16",
        "fresh_subprocess_per_arm": True,
        "shape": {
            "batch_size": args.batch_size, "sequence_length": args.sequence_length,
            "n_embd": args.n_embd, "n_layer": args.n_layer, "n_head": args.n_head,
            "mlp_internal_dim_multiplier": args.mlp_internal_dim_multiplier,
            "block_size": args.block_size, "top_k": args.top_k,
        },
        "warmup": args.warmup,
        "steps": args.steps,
        "arms": arms,
    }
    if "error" not in arms["raw"]:
        for key, arm in arms.items():
            if key != "raw" and "error" not in arm:
                arm["over_raw_throughput_ratio"] = arm["tokens_per_second"] / arms["raw"]["tokens_per_second"]
                arm["over_raw_peak_memory_ratio"] = arm["peak_memory_bytes"] / arms["raw"]["peak_memory_bytes"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
