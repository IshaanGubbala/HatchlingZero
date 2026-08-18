#!/usr/bin/env python3
"""Real-data quality probe: does per-token dynamic block routing
(reference/hz0h_bdh_dynamic_block_routing_layer_torch.py) buy any real
quality advantage over raw (dense) BDH, at any capacity factor -- the
only thing that could justify its real, measured, confirmed speed cost
(docs/restart/hz0h_dynamic_block_routing_cuda_oom_results.md: 1.37x-2.7x
SLOWER than raw at production shape, every capacity factor tested).

Reuses the SAME real data and recipe as this session's own established
Phase F comparison and the FactorizedBDH quality probe
(scripts/hz0h_factorized_quality_probe.py): real 25M-token byte-level
corpus, batch=12, seq=256, AdamW lr=1e-3/weight_decay=0.1, cosine
schedule with warmup, bf16, seed=7 -- so results are directly comparable
to that session's own known baselines (exact BDH 1.582, matched
Transformer 1.738, factorized rank-64 under full curriculum: worse than
both).

**Real, disclosed risk, learned the hard way in this exact session**:
this trains at FIXED depth with NO recurrent-depth curriculum, and runs
far fewer steps than a full Phase F run. FactorizedBDH's own short,
no-curriculum probe found an apparent quality EDGE over dense BDH that
completely reversed once tested under real, full curriculum training
(docs/restart/hz0h_factorized_curriculum_full_comparison_results.md) --
the short-probe result was real but misleading. Any interesting finding
here (positive OR negative) should be treated as preliminary, not
trusted, until verified under the same full curriculum recipe -- not
glossed over as a known risk, actively flagged.

Only the encoder projection is dynamically routed here (encoder_v/
decoder stay dense) -- matches the real, disclosed scope of
`dynamic_block_routing_forward` itself, not a full dynamically-routed
BDH.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_dynamic_block_routing_layer_torch import (
    dynamic_block_routing_forward,
    init_dynamic_block_router,
)
from reference.hz0h_bdh_torch import BDH, BDHConfig


def read_batch(handle, batch_size: int, sequence_length: int, device, epoch_counter: list[int]) -> torch.Tensor:
    values = []
    while len(values) < batch_size:
        line = handle.readline()
        if not line:
            handle.seek(0)
            epoch_counter[0] += 1
            line = handle.readline()
        tokens = json.loads(line)
        if len(tokens) < sequence_length:
            continue
        values.append(tokens[:sequence_length])
    return torch.tensor(np.asarray(values, dtype=np.int64), device=device)


def lr_at_step(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr_ratio: float = 0.1) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return max_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = max_lr * min_lr_ratio
    return min_lr + (max_lr - min_lr) * cosine


def evaluate_raw(model, val_path: Path, batch_size: int, sequence_length: int, device, n_batches: int) -> float:
    model.eval()
    losses = []
    epoch = [0]
    with val_path.open() as handle, torch.no_grad():
        for _ in range(n_batches):
            batch = read_batch(handle, batch_size, sequence_length, device, epoch)
            idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            _, loss = model(idx, targets)
            losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def evaluate_dynamic_routing(model, router, val_path: Path, batch_size: int, sequence_length: int, device,
                              n_batches: int, *, block_size: int, top_k: int, capacity_factor: float) -> tuple[float, float]:
    model.eval()
    losses = []
    drop_rates = []
    epoch = [0]
    with val_path.open() as handle, torch.no_grad():
        for _ in range(n_batches):
            batch = read_batch(handle, batch_size, sequence_length, device, epoch)
            idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            _, loss, routing_per_layer = dynamic_block_routing_forward(
                model, idx, router, targets, block_size=block_size, top_k=top_k, capacity_factor=capacity_factor,
            )
            losses.append(float(loss))
            total_picks = idx.numel() * top_k * model.config.n_head * model.config.n_layer
            total_dropped = sum(r.tokens_dropped for layer in routing_per_layer for r in layer)
            drop_rates.append(total_dropped / total_picks if total_picks else 0.0)
    model.train()
    return sum(losses) / len(losses), sum(drop_rates) / len(drop_rates)


def train_raw(name: str, model, train_path: Path, val_path: Path, *, batch_size: int, sequence_length: int,
              steps: int, warmup_steps: int, max_lr: float, eval_every: int, eval_batches: int, device) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)
    epoch = [0]
    losses = []
    best_val = float("inf")
    val_history = []
    started = time.perf_counter()
    with train_path.open() as handle:
        for step in range(steps):
            lr = lr_at_step(step, steps, warmup_steps, max_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = read_batch(handle, batch_size, sequence_length, device, epoch)
            idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(idx, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if (step + 1) % eval_every == 0 or step == steps - 1:
                val_loss = evaluate_raw(model, val_path, batch_size, sequence_length, device, eval_batches)
                val_history.append({"step": step + 1, "val_loss": val_loss})
                best_val = min(best_val, val_loss)
    elapsed = time.perf_counter() - started
    return {
        "name": name,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "final_train_loss_mean_last_20": sum(losses[-20:]) / len(losses[-20:]),
        "best_validation_loss": best_val,
        "final_validation_loss": val_history[-1]["val_loss"] if val_history else None,
        "validation_history": val_history,
        "training_seconds": elapsed,
        "epochs_of_train_data": epoch[0],
    }


def train_dynamic_routing(name: str, model, router, train_path: Path, val_path: Path, *, batch_size: int,
                           sequence_length: int, steps: int, warmup_steps: int, max_lr: float, eval_every: int,
                           eval_batches: int, device, block_size: int, top_k: int, capacity_factor: float) -> dict:
    optimizer = torch.optim.AdamW(list(model.parameters()) + [router], lr=max_lr, weight_decay=0.1)
    epoch = [0]
    losses = []
    train_drop_rates = []
    best_val = float("inf")
    val_history = []
    started = time.perf_counter()
    with train_path.open() as handle:
        for step in range(steps):
            lr = lr_at_step(step, steps, warmup_steps, max_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = read_batch(handle, batch_size, sequence_length, device, epoch)
            idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            optimizer.zero_grad(set_to_none=True)
            _, loss, routing_per_layer = dynamic_block_routing_forward(
                model, idx, router, targets, block_size=block_size, top_k=top_k, capacity_factor=capacity_factor,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            total_picks = idx.numel() * top_k * model.config.n_head * model.config.n_layer
            total_dropped = sum(r.tokens_dropped for layer in routing_per_layer for r in layer)
            train_drop_rates.append(total_dropped / total_picks if total_picks else 0.0)
            if (step + 1) % eval_every == 0 or step == steps - 1:
                val_loss, val_drop_rate = evaluate_dynamic_routing(
                    model, router, val_path, batch_size, sequence_length, device, eval_batches,
                    block_size=block_size, top_k=top_k, capacity_factor=capacity_factor,
                )
                val_history.append({"step": step + 1, "val_loss": val_loss, "val_drop_rate": val_drop_rate})
                best_val = min(best_val, val_loss)
    elapsed = time.perf_counter() - started
    return {
        "name": name,
        "parameter_count": sum(p.numel() for p in model.parameters()) + router.numel(),
        "final_train_loss_mean_last_20": sum(losses[-20:]) / len(losses[-20:]),
        "mean_train_drop_rate": sum(train_drop_rates) / len(train_drop_rates),
        "best_validation_loss": best_val,
        "final_validation_loss": val_history[-1]["val_loss"] if val_history else None,
        "validation_history": val_history,
        "training_seconds": elapsed,
        "epochs_of_train_data": epoch[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--capacity-factor", type=float, default=1.0, help="1.0 was the CUDA speed sweep's balanced middle setting: 0.573x raw speed, 23.5% real drop rate.")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500, help="Real, disclosed: much shorter than a full curriculum run -- preliminary signal only, see this script's own docstring for the real risk of trusting a short, no-curriculum result.")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires real CUDA hardware")
    device = torch.device("cuda")
    dtype = torch.bfloat16

    bdh_config = BDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
        vocab_size=args.vocab_size, dropout=0.0,
    )
    n_blocks = (args.n_embd * args.mlp_internal_dim_multiplier // args.n_head) // args.block_size

    common = dict(
        train_path=args.data, val_path=args.validation_data, batch_size=args.batch_size,
        sequence_length=args.sequence_length, steps=args.steps, warmup_steps=args.warmup_steps,
        max_lr=args.learning_rate, eval_every=args.eval_every, eval_batches=args.eval_batches, device=device,
    )

    torch.manual_seed(args.seed)
    raw_model = BDH(bdh_config).to(device=device, dtype=dtype)
    raw_model.attn.freqs = raw_model.attn.freqs.to(torch.float32)
    raw_result = train_raw("raw_bdh", raw_model, **common)
    raw_params = raw_result["parameter_count"]
    del raw_model
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    routed_model = BDH(bdh_config).to(device=device, dtype=dtype)
    routed_model.attn.freqs = routed_model.attn.freqs.to(torch.float32)
    router = init_dynamic_block_router(
        args.n_head, args.n_embd, n_blocks, generator=torch.Generator().manual_seed(args.seed),
    ).to(device=device, dtype=dtype)
    router.requires_grad_(True)
    routed_result = train_dynamic_routing(
        f"dynamic_routing_cf_{args.capacity_factor}", routed_model, router, **common,
        block_size=args.block_size, top_k=args.top_k, capacity_factor=args.capacity_factor,
    )
    del routed_model, router
    torch.cuda.empty_cache()

    results = {
        "device": "cuda",
        "hardware_id": torch.cuda.get_device_name(device),
        "dtype": "bfloat16",
        "data": str(args.data),
        "validation_data": str(args.validation_data),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "block_size": args.block_size,
        "top_k": args.top_k,
        "capacity_factor": args.capacity_factor,
        "raw_bdh": raw_result,
        "dynamic_routing": routed_result,
        "dynamic_routing_vs_raw_bdh_param_ratio": routed_result["parameter_count"] / raw_params,
        "dynamic_routing_best_val_minus_raw_best_val": routed_result["best_validation_loss"] - raw_result["best_validation_loss"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
