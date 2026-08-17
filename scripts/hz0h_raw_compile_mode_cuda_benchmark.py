#!/usr/bin/env python3
"""Isolated CUDA sweep of exact-BDH torch.compile execution modes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig


MODES = ("default", "reduce-overhead", "max-autotune")


def _run_arm(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    config = BDHConfig(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
        vocab_size=256,
        dropout=0.0,
    )
    model = BDH(config).to(device=device, dtype=torch.bfloat16)
    model.attn.freqs = model.attn.freqs.to(torch.float32)
    compiled = torch.compile(model, mode=args.mode, fullgraph=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.1, fused=True
    )
    idx = torch.randint(256, (args.batch_size, args.sequence_length), device=device)
    targets = torch.randint(256, idx.shape, device=device)

    def step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        _logits, loss = compiled(idx, targets)
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
    finite = bool(torch.isfinite(torch.tensor(losses)).all()) and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    return {
        "mode": args.mode,
        "tokens_per_second": idx.numel() * args.steps / seconds,
        "milliseconds_per_step": seconds * 1000.0 / args.steps,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "last_loss": losses[-1],
        "finite": finite,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run-arm", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=MODES, default="default")
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


def _child_command(args: argparse.Namespace, mode: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-arm",
        "--mode",
        mode,
        "--batch-size",
        str(args.batch_size),
        "--sequence-length",
        str(args.sequence_length),
        "--n-embd",
        str(args.n_embd),
        "--n-layer",
        str(args.n_layer),
        "--n-head",
        str(args.n_head),
        "--mlp-internal-dim-multiplier",
        str(args.mlp_internal_dim_multiplier),
        "--warmup",
        str(args.warmup),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
    ]


def main() -> None:
    args = _parse_args()
    if args.run_arm:
        print(json.dumps(_run_arm(args)))
        return
    if args.out is None:
        raise ValueError("--out is required")

    arms = {}
    stderr = {}
    for mode in MODES:
        completed = subprocess.run(
            _child_command(args, mode),
            check=True,
            capture_output=True,
            text=True,
        )
        arms[mode] = json.loads(completed.stdout.strip().splitlines()[-1])
        if completed.stderr.strip():
            stderr[mode] = completed.stderr.strip()

    baseline = arms["default"]
    report = {
        "device": "cuda",
        "dtype": "bfloat16",
        "fresh_subprocess_per_arm": True,
        "shape": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "n_embd": args.n_embd,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "mlp_internal_dim_multiplier": args.mlp_internal_dim_multiplier,
        },
        "warmup": args.warmup,
        "steps": args.steps,
        "arms": arms,
        "throughput_ratio_over_default": {
            mode: arm["tokens_per_second"] / baseline["tokens_per_second"]
            for mode, arm in arms.items()
        },
        "peak_memory_ratio_over_default": {
            mode: arm["peak_memory_bytes"] / baseline["peak_memory_bytes"]
            for mode, arm in arms.items()
        },
        "child_stderr": stderr,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
