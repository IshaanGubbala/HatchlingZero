#!/usr/bin/env python3
"""Isolated CUDA benchmark for persistent, physically packed BlockBDH.

The raw, runtime-gather, and packed arms each run in a fresh subprocess. Block
selection and physical packing happen before compilation and outside timing.
The packed hot path is ordinary dense BDH math at a permanently smaller width.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from reference.hz0h_bdh_blocksparse_torch import bdh_blocksparse_forward
from reference.hz0h_bdh_compiled_blocks_torch import (
    PackedBlockBDH,
    calibrate_compiled_block_layout,
)
from reference.hz0h_bdh_torch import BDH, BDHConfig


ARMS = ("raw", "gather", "packed")


def _fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        payload = parameter.detach().contiguous().view(torch.uint8).cpu().numpy()
        digest.update(payload.tobytes())
    return digest.hexdigest()


def _run_arm(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if args.arm == "raw" and args.active_fraction != 1.0:
        raise ValueError("raw arm requires active_fraction=1.0")

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
    source = BDH(config).to(device=device, dtype=torch.bfloat16).train()
    source.attn.freqs = source.attn.freqs.to(torch.float32)

    # Generate all inputs before PackedBlockBDH construction so constructor
    # allocation cannot alter the random training batches for that arm.
    calibration_batches = [
        torch.randint(
            args.vocab_size,
            (args.calibration_batch_size, args.calibration_sequence_length),
            device=device,
        )
        for _ in range(args.calibration_batches)
    ]
    idx = torch.randint(
        args.vocab_size, (args.batch_size, args.sequence_length), device=device
    )
    targets = torch.randint(args.vocab_size, idx.shape, device=device)

    layout = None
    active_blocks = None
    if args.arm != "raw":
        layout = calibrate_compiled_block_layout(
            source,
            calibration_batches,
            block_size=args.block_size,
            active_fraction=args.active_fraction,
        )
        active_blocks = torch.tensor(layout.block_indices, device=device)

    if args.arm == "packed":
        model = PackedBlockBDH.from_layout(source, layout).train()
        del source
        gc.collect()
        torch.cuda.empty_cache()
    else:
        model = source

    if args.arm == "gather":
        def forward(token_ids, target_ids):
            return bdh_blocksparse_forward(
                model, token_ids, active_blocks, args.block_size, target_ids
            )
    else:
        def forward(token_ids, target_ids):
            return model(token_ids, target_ids)

    compiled_forward = torch.compile(
        forward, mode=args.compile_mode, fullgraph=True
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )

    def step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        _logits, loss = compiled_forward(idx, targets)
        loss.backward()
        optimizer.step()
        return loss.detach()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    losses = []
    started = time.perf_counter()
    for _ in range(args.steps):
        losses.append(float(step()))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    return {
        "arm": args.arm,
        "active_fraction": args.active_fraction,
        "block_size": args.block_size,
        "active_blocks": None if layout is None else list(layout.block_indices),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "steps": args.steps,
        "tokens": idx.numel() * args.steps,
        "seconds": elapsed,
        "milliseconds_per_step": elapsed * 1000.0 / args.steps,
        "tokens_per_second": idx.numel() * args.steps / elapsed,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "last_loss": losses[-1],
        "finite": bool(torch.isfinite(torch.tensor(losses)).all()) and gradients_finite,
        "parameter_fingerprint": _fingerprint(model),
    }


def _child_command(args: argparse.Namespace, arm: str, fraction: float) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-arm",
        "--arm", arm,
        "--active-fraction", str(fraction),
        "--block-size", str(args.block_size),
        "--batch-size", str(args.batch_size),
        "--sequence-length", str(args.sequence_length),
        "--n-embd", str(args.n_embd),
        "--n-layer", str(args.n_layer),
        "--n-head", str(args.n_head),
        "--mlp-internal-dim-multiplier", str(args.mlp_internal_dim_multiplier),
        "--vocab-size", str(args.vocab_size),
        "--calibration-batches", str(args.calibration_batches),
        "--calibration-batch-size", str(args.calibration_batch_size),
        "--calibration-sequence-length", str(args.calibration_sequence_length),
        "--warmup", str(args.warmup),
        "--steps", str(args.steps),
        "--seed", str(args.seed),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--compile-mode", args.compile_mode,
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run-arm", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, default="raw")
    parser.add_argument("--active-fraction", type=float, default=1.0)
    parser.add_argument("--fractions", type=float, nargs="+", default=(0.75, 0.5, 0.25))
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--calibration-batches", type=int, default=2)
    parser.add_argument("--calibration-batch-size", type=int, default=2)
    parser.add_argument("--calibration-sequence-length", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="max-autotune",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.run_arm:
        print(json.dumps(_run_arm(args)))
        return
    if args.out is None:
        raise ValueError("--out is required for the aggregate benchmark")
    if any(not 0 < fraction < 1 for fraction in args.fractions):
        raise ValueError("aggregate --fractions must all be in (0, 1)")

    arms = {}
    child_stderr = {}
    specs = [("raw", 1.0)] + [
        (arm, fraction)
        for fraction in args.fractions
        for arm in ("gather", "packed")
    ]
    for arm, fraction in specs:
        name = arm if arm == "raw" else f"{arm}_{fraction:g}"
        completed = subprocess.run(
            _child_command(args, arm, fraction),
            check=True,
            capture_output=True,
            text=True,
        )
        arms[name] = json.loads(completed.stdout.strip().splitlines()[-1])
        if completed.stderr.strip():
            child_stderr[name] = completed.stderr.strip()

    raw = arms["raw"]
    comparisons = {}
    for fraction in args.fractions:
        gather = arms[f"gather_{fraction:g}"]
        packed = arms[f"packed_{fraction:g}"]
        comparisons[f"fraction_{fraction:g}"] = {
            "packed_over_raw_throughput": packed["tokens_per_second"] / raw["tokens_per_second"],
            "packed_over_gather_throughput": packed["tokens_per_second"] / gather["tokens_per_second"],
            "packed_over_raw_allocated_memory": packed["peak_memory_allocated_bytes"] / raw["peak_memory_allocated_bytes"],
            "packed_over_gather_allocated_memory": packed["peak_memory_allocated_bytes"] / gather["peak_memory_allocated_bytes"],
            "packed_minus_gather_loss": packed["last_loss"] - gather["last_loss"],
            "same_selected_blocks": packed["active_blocks"] == gather["active_blocks"],
        }

    report = {
        "device": "cuda",
        "dtype": "bfloat16",
        "compile_mode": args.compile_mode,
        "fresh_subprocess_per_arm": True,
        "calibration_outside_timing": True,
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
        "arms": arms,
        "comparisons": comparisons,
        "child_stderr": child_stderr,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
