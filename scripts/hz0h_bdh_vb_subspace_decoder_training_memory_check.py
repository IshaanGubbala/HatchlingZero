#!/usr/bin/env python3
"""Real optimizer + activation peak-memory check, exact BDH baseline vs
the compound VB-frozen-identity + subspace-decoder architecture. Every
training-time result so far this session (val_loss, training_seconds)
came from real GPU runs, but none of them recorded peak memory during
training itself -- one of section 17's benchmark-protocol checklist
items ("optimizer + activation memory") that was never measured. Same
real training step as scripts/hz0h_bdh_vb_subspace_decoder_quality_check.py
(gradient checkpointing, AdamW, bf16 autocast) -- this just runs a
handful of real steps and records peak allocated/reserved memory instead
of running the full 5M-token budget (quality is already established
separately; this is purely a memory-footprint measurement).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_checkpointed_torch import bdh_variable_depth_forward_checkpointed
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_vb_subspace_decoder_checkpointed_torch import bdh_vb_subspace_decoder_forward_checkpointed
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_combined_best_comparison import autocast_context, make_optimizer
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import read_batch


def run_steps(model, forward_fn, args, device, n_iterations):
    optimizer = make_optimizer(model.parameters(), args, device)
    epochs = [0]
    with args.data.open() as handle:
        for step in range(args.steps):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args, device):
                _, loss = forward_fn(model, idx, n_iterations, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
    synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()

    device = pick_device(args.device)
    results = {"device": str(device), "dtype": args.dtype,
               "note": "real peak allocated/reserved GPU memory across a handful of real training steps (forward+backward+AdamW optimizer.step, gradient checkpointing), full n_layer depth. Not a quality or throughput claim.",
               "config": {"n_embd": args.n_embd, "mult": args.mult, "n_layer": args.n_layer, "n_head": args.n_head,
                          "d_state": args.d_state, "subspace_rank": args.subspace_rank,
                          "batch_size": args.batch_size, "sequence_length": args.sequence_length, "steps": args.steps}}

    torch.manual_seed(args.seed)
    bdh_config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                            mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0)
    bdh_model = BDH(bdh_config).to(device=device, dtype=torch.float32)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    run_steps(bdh_model, bdh_variable_depth_forward_checkpointed, args, device, args.n_layer)
    results["bdh_baseline"] = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
        "parameter_count": sum(p.numel() for p in bdh_model.parameters()),
    }
    if device.type == "cuda":
        print(f"[bdh] peak_allocated={results['bdh_baseline']['peak_allocated_bytes']/1e9:.3f}GB "
              f"peak_reserved={results['bdh_baseline']['peak_reserved_bytes']/1e9:.3f}GB", flush=True)
    del bdh_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    compound_config = BDHVBSubspaceDecoderConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                                                  mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
                                                  d_state=args.d_state, subspace_rank=args.subspace_rank)
    compound_model = BDHVBSubspaceDecoder(compound_config).to(device=device, dtype=torch.float32)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    run_steps(compound_model, bdh_vb_subspace_decoder_forward_checkpointed, args, device, args.n_layer)
    results["compound"] = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
        "parameter_count": sum(p.numel() for p in compound_model.parameters()),
    }
    if device.type == "cuda":
        print(f"[compound] peak_allocated={results['compound']['peak_allocated_bytes']/1e9:.3f}GB "
              f"peak_reserved={results['compound']['peak_reserved_bytes']/1e9:.3f}GB", flush=True)

    if results["bdh_baseline"]["peak_allocated_bytes"] and results["compound"]["peak_allocated_bytes"]:
        results["compound_peak_allocated_reduction_factor"] = results["bdh_baseline"]["peak_allocated_bytes"] / results["compound"]["peak_allocated_bytes"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
