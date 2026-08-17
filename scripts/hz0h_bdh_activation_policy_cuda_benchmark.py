#!/usr/bin/env python3
"""Isolated CUDA throughput/VRAM benchmark for BDH activation policies.

Each arm runs in a fresh subprocess so compiled graphs, optimizer state, and
allocator state from one policy cannot handicap another policy's cuBLAS
algorithm selection. The benchmark measures complete training steps:
forward, cross-entropy, backward, and fused AdamW.
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

from reference.hz0h_bdh_checkpointed_torch import bdh_training_forward
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _step(model, forward_fn, idx, targets, optimizer) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    _logits, loss = forward_fn(model, idx, targets)
    loss.backward()
    optimizer.step()
    return loss.detach()


def _run_arm(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    config = BDHConfig(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
        vocab_size=args.vocab_size,
        dropout=0.0,
    )
    model = BDH(config).to(device=device, dtype=torch.bfloat16)
    model.attn.freqs = model.attn.freqs.to(torch.float32)
    model.train()
    idx = torch.randint(
        args.vocab_size,
        (args.batch_size, args.sequence_length),
        device=device,
    )
    targets = torch.randint(
        args.vocab_size,
        (args.batch_size, args.sequence_length),
        device=device,
    )

    def selected_forward(model_arg, token_ids, target_ids):
        return bdh_training_forward(
            model_arg,
            token_ids,
            n_iterations=args.n_layer,
            targets=target_ids,
            activation_policy=args.policy,
            checkpoint_segment_size=args.checkpoint_segment_size,
        )

    forward_fn = (
        torch.compile(selected_forward, mode=args.compile_mode)
        if args.compiled
        else selected_forward
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.1,
        fused=True,
    )

    for _ in range(args.warmup):
        _step(model, forward_fn, idx, targets, optimizer)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    losses = []
    started = time.perf_counter()
    for _ in range(args.steps):
        losses.append(float(_step(model, forward_fn, idx, targets, optimizer)))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    return {
        "policy": args.policy,
        "compiled": args.compiled,
        "compile_mode": args.compile_mode if args.compiled else None,
        "checkpoint_segment_size": args.checkpoint_segment_size,
        "steps": args.steps,
        "tokens": idx.numel() * args.steps,
        "seconds": elapsed,
        "milliseconds_per_step": elapsed * 1000.0 / args.steps,
        "tokens_per_second": idx.numel() * args.steps / elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "last_loss": losses[-1],
        "finite": all(torch.isfinite(torch.tensor(loss)) for loss in losses)
        and gradients_finite,
    }


def _child_command(args: argparse.Namespace, policy: str, compiled: bool) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-arm",
        "--policy",
        policy,
        "--checkpoint-segment-size",
        str(args.checkpoint_segment_size),
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
        "--vocab-size",
        str(args.vocab_size),
        "--warmup",
        str(args.warmup),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--learning-rate",
        str(args.learning_rate),
        "--compile-mode",
        args.compile_mode,
    ]
    if compiled:
        command.append("--compiled")
    return command


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run-arm", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--policy", choices=("store", "recompute"), default="store")
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument("--checkpoint-segment-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.checkpoint_segment_size < 1:
        raise ValueError("--checkpoint-segment-size must be at least 1")
    if args.run_arm:
        print(json.dumps(_run_arm(args)))
        return
    if args.out is None:
        raise ValueError("--out is required for the aggregate benchmark")

    arms = {}
    child_stderr = {}
    for policy, compiled in (
        ("store", False),
        ("recompute", False),
        ("store", True),
        ("recompute", True),
    ):
        name = f"{'compiled' if compiled else 'eager'}_{policy}"
        completed = subprocess.run(
            _child_command(args, policy, compiled),
            check=True,
            capture_output=True,
            text=True,
        )
        arms[name] = json.loads(completed.stdout.strip().splitlines()[-1])
        if completed.stderr.strip():
            child_stderr[name] = completed.stderr.strip()

    comparisons = {}
    for execution in ("eager", "compiled"):
        store = arms[f"{execution}_store"]
        recompute = arms[f"{execution}_recompute"]
        comparisons[execution] = {
            "recompute_over_store_throughput_ratio": recompute["tokens_per_second"] / store["tokens_per_second"],
            "recompute_over_store_peak_memory_ratio": recompute["peak_memory_bytes"] / store["peak_memory_bytes"],
            "recompute_minus_store_milliseconds_per_step": recompute["milliseconds_per_step"] - store["milliseconds_per_step"],
        }

    report = {
        "device": "cuda",
        "hardware_id": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "queried in child processes",
        "dtype": "bfloat16",
        "shape": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "n_embd": args.n_embd,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "mlp_internal_dim_multiplier": args.mlp_internal_dim_multiplier,
        },
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "fresh_subprocess_per_arm": True,
        "arms": arms,
        "comparisons": comparisons,
        "child_stderr": child_stderr,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
