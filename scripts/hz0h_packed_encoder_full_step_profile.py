#!/usr/bin/env python3
"""Operator-level CUDA time breakdown for one full packed-encoder
checkpointed training step, at production shape.

Before building a custom fused backward to batch the eight rounds' tied
decoder-weight-gradient GEMMs into one big GEMM (Highest-Value Ideas item
2), this measures how much of a real training step's CUDA time those
specific GEMMs actually consume -- the same "measure before a big custom-
backward investment" discipline that would have caught FlashBDH's real
negative result earlier if applied first there.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_packed_encoder_torch import PackedEncoderBDH, bdh_packed_encoder_forward_checkpointed
from reference.hz0h_bdh_torch import BDHConfig


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    config = BDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.multiplier, vocab_size=args.vocab_size, dropout=0.0,
    )
    model = PackedEncoderBDH(config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    idx = torch.randint(args.vocab_size, (args.batch_size, args.sequence_length), device=device, generator=generator)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.sequence_length), device=device, generator=generator)

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = bdh_packed_encoder_forward_checkpointed(
                model, idx, model.config.n_layer, targets, checkpoint_segment_size=1,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for _ in range(args.warmup):
        one_step()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(args.profiled_steps):
            one_step()
        torch.cuda.synchronize()

    def self_cuda_us(e):
        return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)

    key_averages = prof.key_averages(group_by_input_shape=True)
    total_cuda_us = sum(max(self_cuda_us(e), 0) for e in key_averages)

    rows = []
    for e in key_averages:
        t = self_cuda_us(e)
        if t <= 0:
            continue
        rows.append({
            "name": e.key,
            "input_shapes": str(e.input_shapes),
            "count": e.count,
            "self_cuda_us": t,
            "self_cuda_pct": 100.0 * t / total_cuda_us if total_cuda_us else 0.0,
        })
    rows.sort(key=lambda r: -r["self_cuda_us"])

    D = args.n_embd
    nh = args.n_head
    N = D * args.multiplier // nh
    wide = nh * N
    decoder_grad_shapes = {f"[[{wide}, {args.batch_size * args.sequence_length}], [{args.batch_size * args.sequence_length}, {D}]]",
                            f"[[{args.batch_size * args.sequence_length}, {wide}], [{args.batch_size * args.sequence_length}, {D}]]"}

    report = {
        "config": {
            "batch_size": args.batch_size, "sequence_length": args.sequence_length,
            "n_embd": D, "n_head": nh, "multiplier": args.multiplier, "n_layer": args.n_layer,
            "decoder_shape": [wide, D],
        },
        "total_self_cuda_us_all_ops": total_cuda_us,
        "profiled_steps": args.profiled_steps,
        "top_20_ops_by_self_cuda_time": rows[:20],
        "matmul_like_ops": [r for r in rows if any(t in r["name"].lower() for t in ("mm", "bmm", "gemm", "matmul", "addmm"))][:30],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--multiplier", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profiled-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
