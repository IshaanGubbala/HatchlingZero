#!/usr/bin/env python3
"""Real-data quality probe for FactorizedBDH: does the codex-mac CUDA
architecture sweep's speed/memory win (docs/restart/hz0h_factorized_*
results) come at a quality cost relative to raw (dense) BDH?

That sweep used synthetic random tokens -- fine for measuring throughput,
meaningless for quality (no structure to learn, so any two models look
the same regardless of architecture). This script reuses the SAME real
data and recipe as this session's own established Phase F comparison
(docs/restart/hz0h_phase_f_same_gpu_comparison_results.md: real 25M-token
byte-level corpus, batch=12, seq=256, AdamW lr=1e-3/weight_decay=0.1,
cosine schedule with warmup, bf16, seed=7) so the resulting validation
loss numbers are directly comparable to that session's own known
baseline (exact BDH 1.582, matched Transformer 1.738).

Real, disclosed scope limit: this trains all arms at FIXED depth
(--n-layer, no recurrent-depth curriculum), unlike Phase F's own
curriculum-boosted exact-BDH arm -- FactorizedBDH has no curriculum-
compatible forward built yet, and mixing a curriculum-boosted control
against a non-curriculum candidate would confound the comparison this
script exists to make (factorization's OWN quality cost vs dense BDH,
same depth, same everything else). Also real and disclosed: the default
step count here is much shorter than a full Phase F run, to get a real
but preliminary signal without committing hours of GPU time -- report
the real numbers, don't imply this reproduces the full 1.582/1.738
headline result.
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

from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0i_bdh_model import HZ0IBDHConfig
from reference.hz0i_factorized_bdh import FactorizedBDH


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


def evaluate(model, val_path: Path, batch_size: int, sequence_length: int, device, n_batches: int, *, transformer: bool) -> float:
    model.eval()
    losses = []
    epoch = [0]
    with val_path.open() as handle, torch.no_grad():
        for _ in range(n_batches):
            batch = read_batch(handle, batch_size, sequence_length, device, epoch)
            idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            if transformer:
                logits = model(idx)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            else:
                _, loss = model(idx, targets)
            losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def train_one(name: str, model, train_path: Path, val_path: Path, *, batch_size: int, sequence_length: int,
              steps: int, warmup_steps: int, max_lr: float, eval_every: int, eval_batches: int,
              device, transformer: bool) -> dict:
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
            if transformer:
                logits = model(idx)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            else:
                _, loss = model(idx, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if (step + 1) % eval_every == 0 or step == steps - 1:
                val_loss = evaluate(model, val_path, batch_size, sequence_length, device, eval_batches, transformer=transformer)
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
    parser.add_argument("--rank", type=int, default=64, help="FactorizedBDH's rank; 64 was the standout speed/memory winner in the CUDA architecture sweep.")
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-d-ff", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500, help="Real, disclosed: much shorter than Phase F's own full run -- a preliminary signal, not a reproduction of the 1.582/1.738 headline numbers.")
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
    factor_config = HZ0IBDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
        vocab_size=args.vocab_size, dropout=0.0,
    )
    transformer_config = MatchedTransformerConfig({
        "vocab_size": args.vocab_size, "d_model": args.n_embd,
        "num_layers": args.transformer_layers, "num_heads": args.transformer_heads,
        "head_dim": args.n_embd // args.transformer_heads, "d_ff": args.transformer_d_ff,
        "use_rope": True,
    })

    common = dict(
        train_path=args.data, val_path=args.validation_data, batch_size=args.batch_size,
        sequence_length=args.sequence_length, steps=args.steps, warmup_steps=args.warmup_steps,
        max_lr=args.learning_rate, eval_every=args.eval_every, eval_batches=args.eval_batches, device=device,
    )

    torch.manual_seed(args.seed)
    raw_model = BDH(bdh_config).to(device=device, dtype=dtype)
    raw_result = train_one("raw_bdh", raw_model, **common, transformer=False)
    raw_params = raw_result["parameter_count"]
    del raw_model
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    factorized_model = FactorizedBDH(factor_config, rank=args.rank).to(device=device, dtype=dtype)
    factorized_result = train_one(f"factorized_rank_{args.rank}", factorized_model, **common, transformer=False)
    del factorized_model
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    transformer_model = MatchedTransformerLM(transformer_config).to(device=device, dtype=dtype)
    transformer_result = train_one("matched_transformer", transformer_model, **common, transformer=True)
    del transformer_model
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
        "rank": args.rank,
        "raw_bdh": raw_result,
        "factorized": factorized_result,
        "matched_transformer": transformer_result,
        "factorized_vs_raw_bdh_param_ratio": factorized_result["parameter_count"] / raw_params,
        "factorized_best_val_minus_raw_best_val": factorized_result["best_validation_loss"] - raw_result["best_validation_loss"],
        "raw_bdh_best_val_minus_transformer_best_val": raw_result["best_validation_loss"] - transformer_result["best_validation_loss"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
