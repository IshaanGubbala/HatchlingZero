#!/usr/bin/env python3
"""CUDA-only raw-vs-chunk_gla systems preflight for sparse Direct Split-V.

This is deliberately a fixed-route, untrained kernel measurement. It proves
neither language quality nor BlockBDH-vs-Transformer superiority; output is
always marked claim_eligible=false.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from reference.hz0h_bdh_blocksparse_torch import (
    bdh_blocksparse_direct_split_v_chunk_gla_forward,
    bdh_blocksparse_direct_split_v_forward,
    compute_active_blocks,
)
from reference.hz0h_bdh_torch import BDH, BDHConfig


def make_model(config: BDHConfig, state: dict[str, torch.Tensor], dtype: torch.dtype) -> BDH:
    model = BDH(config).to(device="cuda", dtype=dtype)
    model.attn.freqs = model.attn.freqs.float()
    model.load_state_dict(state)
    return model


def measure(config, state, tokens, targets, active, dtype, fused: bool, warmup: int, steps: int) -> dict:
    model = make_model(config, state, dtype).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    forward = bdh_blocksparse_direct_split_v_chunk_gla_forward if fused else bdh_blocksparse_direct_split_v_forward
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        _, loss = forward(model, tokens, active, block_size=16, targets=targets)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = forward(model, tokens, active, block_size=16, targets=targets)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "seconds_per_step": elapsed / steps,
        "tokens_per_second": tokens.numel() * steps / elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "last_loss": float(loss.detach()),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_gradients": all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
    }
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=1)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--active-fraction", type=float, default=0.03125)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this preflight requires CUDA and flash-linear-attention/Triton")
    if args.block_size <= 0 or args.block_size % 2:
        raise ValueError("block size must be positive and even")
    config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size, dropout=0.0)
    latent_width = args.n_embd * args.mlp_internal_dim_multiplier // args.n_head
    if latent_width % args.block_size:
        raise ValueError("block size must divide latent width")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    initial = BDH(config).to(device="cuda", dtype=dtype)
    initial.attn.freqs = initial.attn.freqs.float()
    state = {key: value.detach().cpu().clone() for key, value in initial.state_dict().items()}
    del initial
    torch.cuda.empty_cache()
    tokens = torch.randint(0, args.vocab_size, (args.batch_size, args.sequence_length), device="cuda")
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.sequence_length), device="cuda")
    route_model = make_model(config, state, dtype)
    active = compute_active_blocks(route_model, tokens, args.block_size, args.active_fraction, method="cheap_proxy")
    del route_model
    torch.cuda.empty_cache()
    # Check exact sparse composition before timing separate clean models.
    raw_model = make_model(config, state, dtype).eval()
    fused_model = make_model(config, state, dtype).eval()
    with torch.inference_mode():
        raw_logits, raw_loss = bdh_blocksparse_direct_split_v_forward(raw_model, tokens, active, args.block_size, targets=targets)
        fused_logits, fused_loss = bdh_blocksparse_direct_split_v_chunk_gla_forward(fused_model, tokens, active, args.block_size, targets=targets)
    max_logit_difference = float((raw_logits - fused_logits).abs().max())
    loss_difference = float((raw_loss - fused_loss).abs())
    del raw_model, fused_model, raw_logits, fused_logits, raw_loss, fused_loss
    torch.cuda.empty_cache()
    raw = measure(config, state, tokens, targets, active, dtype, False, args.warmup, args.steps)
    fused = measure(config, state, tokens, targets, active, dtype, True, args.warmup, args.steps)
    report = {
        "architecture": "block_bdh_direct_split_v_chunk_gla_derivative",
        "exact_bdh": False, "trained_weights": False, "claim_eligible": False,
        "device": "cuda", "hardware_id": torch.cuda.get_device_name(), "dtype": "bfloat16",
        "batch_size": args.batch_size, "sequence_length": args.sequence_length,
        "effective_batch_tokens": args.batch_size * args.sequence_length,
        "n_embd": args.n_embd, "n_layer": args.n_layer, "n_head": args.n_head,
        "mlp_internal_dim_multiplier": args.mlp_internal_dim_multiplier,
        "parameter_count": sum(value.numel() for value in state.values()),
        "block_size": args.block_size, "active_fraction": args.active_fraction,
        "router_method": "cheap_proxy", "route_indices": [int(v) for v in active.cpu().tolist()],
        "optimizer": "AdamW fused", "compile_step": False,
        "numerical_preflight": {"max_logit_difference": max_logit_difference, "loss_difference": loss_difference},
        "raw": raw, "chunk_gla": fused,
        "chunk_gla_over_raw_speed_ratio": fused["tokens_per_second"] / raw["tokens_per_second"],
        "chunk_gla_over_raw_peak_memory_ratio": fused["peak_memory_bytes"] / raw["peak_memory_bytes"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
