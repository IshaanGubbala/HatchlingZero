#!/usr/bin/env python3
"""Tier 3 item 17's real wall-clock test: does the correctness-verified
batched exact x-skip (reference/hz0h_bdh_exact_x_skip_batched_torch.py)
actually run faster than dense at production shape on real GPU hardware,
or does it lose to gather/indexing overhead the way several other
mathematically-sound sparse ideas did earlier this session (plan section
3, Constraint D)? This is the deciding measurement -- correctness alone
(item 16/17's earlier checks) does not justify further investment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_exact_x_skip_torch import bdh_round_dense
from reference.hz0h_bdh_exact_x_skip_batched_torch import bdh_round_exact_x_skip_batched


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_rounds(fn, x0, enc, enc_v, dec, attn, ln, config, n_rounds: int, device: torch.device, repeats: int) -> dict:
    with torch.no_grad():
        x = x0
        for _ in range(n_rounds):
            x = fn(x, enc, enc_v, dec, attn, ln, config)  # warmup
        _sync(device)
        started = time.perf_counter()
        for _ in range(repeats):
            x = x0
            for _ in range(n_rounds):
                x = fn(x, enc, enc_v, dec, attn, ln, config)
        _sync(device)
        elapsed = time.perf_counter() - started
    return {"seconds_per_repeat": elapsed / repeats, "elapsed_total": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", type=Path, default=None,
                         help="Load real trained weights instead of random init -- random-init BDH weights "
                              "produce ~50% ReLU density (symmetric random projection), NOT representative of "
                              "a trained model's real sparsity (~28% at production geometry per this session's "
                              "own x-sparsity diagnostic). Without this flag, results describe an untrained "
                              "execution-speed diagnostic only.")
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

    idx = torch.randint(0, args.vocab_size, (args.batch_size, args.sequence_length), device=device)
    x0 = model.ln(model.embed(idx).unsqueeze(1))
    enc = model._w(model.encoder)
    enc_v = model._w(model.encoder_v)
    dec = model._w(model.decoder)

    print(f"[bench] shape B={args.batch_size} T={args.sequence_length} D={args.n_embd} nh={args.n_head} "
          f"N={args.n_embd*args.mult//args.n_head} n_layer={args.n_layer} dtype={args.dtype}", flush=True)

    dense = time_rounds(bdh_round_dense, x0, enc, enc_v, dec, model.attn, model.ln, config, args.n_layer, device, args.repeats)
    print(f"[bench] dense: {dense['seconds_per_repeat']*1000:.2f} ms/repeat ({args.n_layer} rounds)", flush=True)

    batched = time_rounds(bdh_round_exact_x_skip_batched, x0, enc, enc_v, dec, model.attn, model.ln, config, args.n_layer, device, args.repeats)
    print(f"[bench] batched exact-skip: {batched['seconds_per_repeat']*1000:.2f} ms/repeat ({args.n_layer} rounds)", flush=True)

    speedup = dense["seconds_per_repeat"] / batched["seconds_per_repeat"]
    print(f"[bench] speedup (dense / batched): {speedup:.3f}x", flush=True)

    report = {
        "config": {"n_embd": args.n_embd, "n_head": args.n_head, "mult": args.mult, "n_layer": args.n_layer,
                    "batch_size": args.batch_size, "sequence_length": args.sequence_length, "dtype": args.dtype},
        "dense": dense, "batched_exact_skip": batched, "speedup_dense_over_batched": speedup,
        "note": "real wall-clock test of the correctness-verified batched exact x-skip vs dense, at production shape. "
                 "Oracle-packed ceiling (pre-gathered indices outside the timed region) was 3.33x -- this measures "
                 "the REAL cost including the gather/argsort/scatter overhead this implementation actually pays.",
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
