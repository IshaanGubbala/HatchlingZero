#!/usr/bin/env python3
"""Tier 3 item 20's real wall-clock test: does the correctness-verified
batched sparse-state-row kernel (reference/hz0h_bdh_sparse_state_row_kernel_torch.py)
actually decode faster than dense bdh_stream_chunk at realistic serving
batch sizes on real GPU hardware? The oracle ceiling (2.101x) and item
17's 126x real-world loss for a structurally similar approach both bear
on the verdict here -- this is the deciding measurement, not the ceiling
alone.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig, init_bdh_states, bdh_stream_chunk
from reference.hz0h_bdh_sparse_state_row_kernel_torch import bdh_stream_step_sparse_row_batched


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_decode(fn, model, tokens, device, warmup: int, repeats: int) -> dict:
    B, T = tokens.shape
    with torch.no_grad():
        states = init_bdh_states(model, B, device, next(model.parameters()).dtype)
        for pos in range(min(warmup, T)):
            states, _ = fn(model, states, tokens[:, pos:pos + 1], pos)
        _sync(device)
        states = init_bdh_states(model, B, device, next(model.parameters()).dtype)
        started = time.perf_counter()
        for pos in range(min(repeats, T)):
            states, _ = fn(model, states, tokens[:, pos:pos + 1], pos)
        _sync(device)
        elapsed = time.perf_counter() - started
    return {"seconds_per_step": elapsed / min(repeats, T)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--batch-sizes", type=str, default="1,8,32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    torch.manual_seed(args.seed)

    config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                        mlp_internal_dim_multiplier=args.mult, vocab_size=args.vocab_size, dropout=0.0)
    model = BDH(config).to(device=device, dtype=torch.float32).eval()
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(payload["state_dict"])
        print(f"[bench] loaded real trained weights from {args.checkpoint}", flush=True)
    model = model.to(dtype=torch_dtype)
    model.attn.freqs = model.attn.freqs.to(torch.float32)

    results = {}
    for B in (int(b) for b in args.batch_sizes.split(",")):
        tokens = torch.randint(0, args.vocab_size, (B, args.warmup + args.repeats), device=device)
        dense = bench_decode(bdh_stream_chunk, model, tokens, device, args.warmup, args.repeats)
        batched = bench_decode(bdh_stream_step_sparse_row_batched, model, tokens, device, args.warmup, args.repeats)
        speedup = dense["seconds_per_step"] / batched["seconds_per_step"]
        print(f"[bench] B={B}: dense {dense['seconds_per_step']*1e3:.3f} ms/step | "
              f"sparse-row {batched['seconds_per_step']*1e3:.3f} ms/step | speedup {speedup:.3f}x", flush=True)
        results[B] = {"dense": dense, "sparse_row_batched": batched, "speedup": speedup}

    report = {"config": {"n_embd": args.n_embd, "n_head": args.n_head, "mult": args.mult, "n_layer": args.n_layer, "dtype": args.dtype}, "by_batch_size": results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
