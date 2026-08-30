#!/usr/bin/env python3
"""Real local (MPS/CPU, NOT the CUDA numbers the quality-check dispatch
reports) wall-clock comparison: base compound BDHVBSubspaceDecoder
(n_layer rounds, every round re-addresses -- the architecture BDH-Delta's
addressing pipeline was extracted from) vs BDH-Delta (K exact refreshes
x M cheap think-steps), at matched production dims. Tests both:

  - training-step-shaped cost: forward+backward, matched total think
    depth (n_layer == n_refresh*n_think) so both arms do the same
    number of recurrent state updates, differing only in how many of
    those updates re-run the EXPENSIVE exact-addressing pipeline.
  - naive (no-KV-cache) autoregressive decode: BDHVB.generate's own
    pattern (full forward recomputed every new token) vs
    generate_with_carry's same recompute-every-token pattern -- an
    apples-to-apples comparison of PER-TOKEN cost, not a claim about
    either implementation's best-case cached-decode speed (neither
    variant benchmarked here uses this project's separate streaming/
    KV-cache decode path).

This is a real, honest LOCAL signal for whether decoupling refresh
cadence from think depth is winning on wall-clock the way the
plan's central claim (section 2/9) predicts -- not a substitute for
the real CUDA quality-check number, which measures val_loss, not speed.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_delta_vnext_torch import add_delta_vnext, bdh_delta_vnext_forward, generate_with_carry
from reference.hz0h_bdh_vb_subspace_decoder_checkpointed_torch import bdh_vb_subspace_decoder_forward_checkpointed
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize


def timed(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    synchronize(device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), max(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=8, help="base model round count AND delta's n_refresh*n_think target")
    parser.add_argument("--n-refresh", type=int, default=4)
    parser.add_argument("--n-think", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--decode-tokens", type=int, default=8)
    args = parser.parse_args()

    assert args.n_refresh * args.n_think == args.n_layer, \
        f"n_refresh*n_think ({args.n_refresh*args.n_think}) must equal n_layer ({args.n_layer}) for a matched-think-depth comparison"

    device = pick_device(args.device)
    dtype = torch.float32  # MPS/CPU local, no autocast -- matches this project's other local-only benchmarks
    print(f"[bench] device={device} dtype={dtype} batch={args.batch_size} seq={args.sequence_length} "
          f"n_embd={args.n_embd} n_layer={args.n_layer} n_refresh={args.n_refresh} n_think={args.n_think}", flush=True)

    torch.manual_seed(0)
    config = BDHVBSubspaceDecoderConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
        d_state=args.d_state, subspace_rank=args.subspace_rank,
    )

    base_model = BDHVBSubspaceDecoder(config).to(device=device, dtype=dtype)
    delta_model = BDHVBSubspaceDecoder(config).to(device=device, dtype=dtype)
    add_delta_vnext(delta_model, n_refresh=args.n_refresh, n_think=args.n_think)
    delta_model = delta_model.to(device=device, dtype=dtype)

    base_params = sum(p.numel() for p in base_model.parameters())
    delta_params = sum(p.numel() for p in delta_model.parameters())
    print(f"[bench] base_params={base_params/1e6:.2f}M delta_params={delta_params/1e6:.2f}M "
          f"(delta adds {(delta_params-base_params)/1e6:.2f}M for Think Cell + belief cell + bridges)", flush=True)

    idx = torch.randint(0, 256, (args.batch_size, args.sequence_length), device=device)
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()

    def base_fwd():
        base_model.zero_grad(set_to_none=True)
        _, loss = base_model(x, y)
        return loss

    def base_fwd_bwd():
        base_model.zero_grad(set_to_none=True)
        # Real training (every quality-check script in this project, this
        # arm's own training script included) always checkpoints for
        # memory reasons at this scale -- use the SAME checkpointed
        # forward here, not the plain one, or the base arm would win on
        # activation-memory footprint alone rather than on compute this
        # benchmark is trying to isolate.
        _, loss = bdh_vb_subspace_decoder_forward_checkpointed(base_model, x, args.n_layer, y)
        loss.backward()

    def delta_fwd():
        delta_model.zero_grad(set_to_none=True)
        _, loss = bdh_delta_vnext_forward(delta_model, x, args.n_refresh, y)
        return loss

    def delta_fwd_bwd():
        loss = delta_fwd()
        loss.backward()

    print("\n=== forward only (inference-shaped) ===", flush=True)
    base_med, base_min, base_max = timed(base_fwd, device, args.warmup, args.iters)
    delta_med, delta_min, delta_max = timed(delta_fwd, device, args.warmup, args.iters)
    print(f"[bench] base  forward: median={base_med*1000:.1f}ms min={base_min*1000:.1f}ms max={base_max*1000:.1f}ms "
          f"({args.batch_size*args.sequence_length/base_med:.0f} tok/s)", flush=True)
    print(f"[bench] delta forward: median={delta_med*1000:.1f}ms min={delta_min*1000:.1f}ms max={delta_max*1000:.1f}ms "
          f"({args.batch_size*args.sequence_length/delta_med:.0f} tok/s)", flush=True)
    print(f"[bench] forward speedup (base/delta) = {base_med/delta_med:.2f}x", flush=True)

    print("\n=== forward+backward (training-step-shaped) ===", flush=True)
    base_med2, base_min2, base_max2 = timed(base_fwd_bwd, device, args.warmup, args.iters)
    delta_med2, delta_min2, delta_max2 = timed(delta_fwd_bwd, device, args.warmup, args.iters)
    print(f"[bench] base  fwd+bwd: median={base_med2*1000:.1f}ms min={base_min2*1000:.1f}ms max={base_max2*1000:.1f}ms "
          f"({args.batch_size*args.sequence_length/base_med2:.0f} tok/s)", flush=True)
    print(f"[bench] delta fwd+bwd: median={delta_med2*1000:.1f}ms min={delta_min2*1000:.1f}ms max={delta_max2*1000:.1f}ms "
          f"({args.batch_size*args.sequence_length/delta_med2:.0f} tok/s)", flush=True)
    print(f"[bench] fwd+bwd speedup (base/delta) = {base_med2/delta_med2:.2f}x", flush=True)

    print(f"\n=== naive autoregressive decode, {args.decode_tokens} new tokens, batch={args.batch_size}, no KV cache ===", flush=True)
    prompt = torch.randint(0, 256, (args.batch_size, 8), device=device)

    def base_decode():
        with torch.no_grad():
            base_model.generate(prompt, max_new_tokens=args.decode_tokens)

    def delta_decode():
        generate_with_carry(delta_model, prompt, max_new_tokens=args.decode_tokens, n_refresh=args.n_refresh)

    base_med3, base_min3, base_max3 = timed(base_decode, device, 1, max(2, args.iters // 2))
    delta_med3, delta_min3, delta_max3 = timed(delta_decode, device, 1, max(2, args.iters // 2))
    print(f"[bench] base  decode: median={base_med3*1000:.1f}ms ({base_med3*1000/args.decode_tokens:.1f}ms/token)", flush=True)
    print(f"[bench] delta decode: median={delta_med3*1000:.1f}ms ({delta_med3*1000/args.decode_tokens:.1f}ms/token)", flush=True)
    print(f"[bench] decode speedup (base/delta) = {base_med3/delta_med3:.2f}x", flush=True)

    print("\n[bench] DONE -- local MPS/CPU numbers only, not a substitute for the CUDA quality-check run", flush=True)


if __name__ == "__main__":
    main()
