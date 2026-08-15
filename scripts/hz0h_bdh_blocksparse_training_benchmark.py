#!/usr/bin/env python3
"""Measure dense BDH vs real BlockBDH training-step execution.

This is a systems probe, not a quality claim: BlockBDH is an explicit
architecture derivative and must later be trained on real language data with
matched validation before promotion.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from reference.hz0h_bdh_blocksparse_torch import bdh_blocksparse_forward, compute_active_blocks
from reference.hz0h_bdh_torch import BDH, BDHConfig


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    return None


def run_variant(model: BDH, tokens: torch.Tensor, targets: torch.Tensor, *, sparse: bool, block_size: int, active_fraction: float, warmup: int, steps: int, device: torch.device) -> dict:
    model.train()
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        if sparse:
            active = compute_active_blocks(model, tokens, block_size=block_size, active_fraction=active_fraction)
            _, loss = bdh_blocksparse_forward(model, tokens, active, block_size=block_size, targets=targets)
        else:
            _, loss = model(tokens, targets=targets)
        loss.backward()
        sync(device)
    reset_peak(device); sync(device); started = time.perf_counter()
    last_loss = None
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        if sparse:
            active = compute_active_blocks(model, tokens, block_size=block_size, active_fraction=active_fraction)
            _, loss = bdh_blocksparse_forward(model, tokens, active, block_size=block_size, targets=targets)
        else:
            _, loss = model(tokens, targets=targets)
        loss.backward()
        last_loss = float(loss.detach().cpu())
        sync(device)
    elapsed = time.perf_counter() - started
    return {"steps": steps, "tokens_per_second": float(tokens.numel() * steps / elapsed), "seconds_per_step": elapsed / steps, "last_loss": last_loss, "peak_memory_bytes": peak_memory(device), "active_fraction": active_fraction if sparse else 1.0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--active-fraction", type=float, default=0.5)
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available(): raise RuntimeError("MPS unavailable")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    torch.manual_seed(7)
    cfg = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head, mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size, dropout=0.0)
    dense = BDH(cfg).to(device=device, dtype=dtype)
    sparse = BDH(cfg).to(device=device, dtype=dtype)
    sparse.load_state_dict(dense.state_dict())
    dense.attn.freqs = dense.attn.freqs.float(); sparse.attn.freqs = sparse.attn.freqs.float()
    tokens = torch.randint(0, args.vocab_size, (args.batch_size, args.sequence_length), device=device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.sequence_length), device=device)
    result = {"architecture": "BlockBDH_training_probe", "device": str(device), "dtype": args.dtype, "parameter_count": sum(p.numel() for p in dense.parameters()), "batch_size": args.batch_size, "sequence_length": args.sequence_length, "block_size": args.block_size, "active_fraction": args.active_fraction, "trained_weights": False, "dense": run_variant(dense, tokens, targets, sparse=False, block_size=args.block_size, active_fraction=args.active_fraction, warmup=args.warmup, steps=args.steps, device=device), "blocksparse": run_variant(sparse, tokens, targets, sparse=True, block_size=args.block_size, active_fraction=args.active_fraction, warmup=args.warmup, steps=args.steps, device=device)}
    result["speed_ratio_blocksparse_over_dense"] = result["blocksparse"]["tokens_per_second"] / result["dense"]["tokens_per_second"]
    result["claim_eligible"] = False
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.output: args.output.write_text(output + "\n")


if __name__ == "__main__": main()
