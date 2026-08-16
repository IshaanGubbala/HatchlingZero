#!/usr/bin/env python3
"""CUDA training-step sweep for GPU-friendly factorized BDH projections.

This is an architecture experiment, not an exact-BDH parity path: each
factorized candidate replaces the dense per-head encoder/value/decoder
matrices with low-rank products. Every arm is timed in isolation so a second
resident model cannot change cuBLAS algorithm selection for the baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0i_bdh_model import HZ0IBDHConfig
from reference.hz0i_factorized_bdh import FactorizedBDH


def _sync() -> None:
    torch.cuda.synchronize()


def _step(model, idx, targets, optimizer, *, transformer: bool = False) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    if transformer:
        logits = model(idx)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    else:
        _, loss = model(idx, targets)
    loss.backward()
    optimizer.step()
    return loss.detach()


def _benchmark(model, idx, targets, *, warmup: int, steps: int, lr: float, transformer: bool = False) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, fused=True)
    for _ in range(warmup):
        _step(model, idx, targets, optimizer, transformer=transformer)
    _sync()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    for _ in range(steps):
        losses.append(float(_step(model, idx, targets, optimizer, transformer=transformer)))
    _sync()
    elapsed = time.perf_counter() - started
    finite = all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    finite = finite and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    return {
        "steps": steps,
        "tokens": idx.numel() * steps,
        "seconds": elapsed,
        "tokens_per_second": idx.numel() * steps / elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "last_loss": losses[-1],
        "finite": finite,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    parser.add_argument("--ranks", default="64,128,256")
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-d-ff", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this sweep requires real CUDA hardware")
    ranks = [int(value) for value in args.ranks.split(",") if value.strip()]
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    idx = torch.randint(args.vocab_size, (args.batch_size, args.sequence_length), device=device)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.sequence_length), device=device)
    base_config = BDHConfig(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
        vocab_size=args.vocab_size,
        dropout=0.0,
    )
    factor_config = HZ0IBDHConfig(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
        vocab_size=args.vocab_size,
        dropout=0.0,
    )

    def fresh(factory):
        torch.manual_seed(args.seed)
        model = factory().to(device=device, dtype=dtype)
        if hasattr(model, "attn") and hasattr(model.attn, "freqs"):
            model.attn.freqs = model.attn.freqs.to(torch.float32)
        return model

    def run(factory, *, transformer: bool = False) -> dict:
        model = fresh(factory)
        try:
            return _benchmark(
                model, idx, targets, warmup=args.warmup, steps=args.steps,
                lr=args.learning_rate, transformer=transformer,
            )
        finally:
            del model
            torch.cuda.empty_cache()
            _sync()

    raw = run(lambda: BDH(base_config))
    candidates = {
        f"factorized_rank_{rank}": run(lambda rank=rank: FactorizedBDH(factor_config, rank=rank))
        for rank in ranks
    }
    transformer_config = MatchedTransformerConfig({
        "vocab_size": args.vocab_size,
        "d_model": args.n_embd,
        "num_layers": args.transformer_layers,
        "num_heads": args.transformer_heads,
        "head_dim": args.n_embd // args.transformer_heads,
        "d_ff": args.transformer_d_ff,
        "use_rope": True,
    })
    transformer = run(lambda: MatchedTransformerLM(transformer_config), transformer=True)
    results = {
        "device": "cuda",
        "hardware_id": torch.cuda.get_device_name(device),
        "dtype": "bfloat16",
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "n_embd": args.n_embd,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "latent_width": args.n_embd * args.mlp_internal_dim_multiplier // args.n_head,
        "warmup_steps": args.warmup,
        "timed_steps": args.steps,
        "raw_bdh": raw,
        "factorized": candidates,
        "matched_transformer": transformer,
    }
    for name, result in candidates.items():
        result["over_raw_speed_ratio"] = result["tokens_per_second"] / raw["tokens_per_second"]
        result["over_transformer_speed_ratio"] = result["tokens_per_second"] / transformer["tokens_per_second"]
        result["over_raw_peak_memory_ratio"] = result["peak_memory_bytes"] / raw["peak_memory_bytes"]
        result["over_transformer_peak_memory_ratio"] = result["peak_memory_bytes"] / transformer["peak_memory_bytes"]
    results["raw_bdh_over_transformer_speed_ratio"] = raw["tokens_per_second"] / transformer["tokens_per_second"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
