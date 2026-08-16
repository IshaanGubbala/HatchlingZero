#!/usr/bin/env python3
"""Real decomposition of the gpu_native end-to-end slowdown
(docs/restart/hz0h_gpu_native_integration_results.md): the combined
forward (Triton kernel + wide-GEMM encoder + bmm encoder_v) measured
0.636x -- 1.57x SLOWER end-to-end -- despite each remap winning alone in
its own forward-only benchmark. This script times 4 real end-to-end
training steps (forward + backward + optimizer.step(), same production
config used for every benchmark this session) with each remap added one
at a time, to find which addition flips the sign:

  1. raw       -- the unmodified oracle, BDH.forward
  2. triton    -- oracle encoder/encoder_v + Triton attention only
                  (already independently confirmed 1.551x in isolation --
                  this run re-confirms it under the same harness as the
                  other 3 points, for a clean apples-to-apples chain)
  3. +bmm      -- adds the bmm encoder_v remap on top of triton
  4. +wide     -- adds the live wide-GEMM encoder on top of that (the
                  full gpu_native integration -- already know this one
                  is 0.636x; this script exists to see the chain that
                  leads there, not just the two endpoints)

Real, disclosed leading hypothesis (not yet confirmed): the wide-GEMM
encoder's LIVE (non-cached) permute+reshape has to run on both forward
and backward every step, a cost step 4 pays that step 3 does not -- if
that's right, points 1-3 should stay roughly monotonically faster (or at
least not regress sharply) and the big drop should appear specifically
between 3 and 4. If the drop instead appears between 2 and 3, that would
point at the bmm encoder_v remap's own backward cost instead -- report
whichever the real numbers show.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from reference.hz0h_bdh_gpu_native_torch import bdh_gpu_native_forward
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _sync() -> None:
    torch.cuda.synchronize()


def _step(model, idx, targets, *, use_wide_encoder, use_bmm_encoder_v, use_triton_attention, optimizer) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    _, loss = bdh_gpu_native_forward(
        model, idx, targets,
        use_wide_encoder=use_wide_encoder,
        use_bmm_encoder_v=use_bmm_encoder_v,
        use_triton_attention=use_triton_attention,
    )
    loss.backward()
    optimizer.step()
    return loss.detach()


def _benchmark(model, idx, targets, *, use_wide_encoder, use_bmm_encoder_v, use_triton_attention, warmup, steps, lr) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, fused=True)
    for _ in range(warmup):
        _step(model, idx, targets, use_wide_encoder=use_wide_encoder, use_bmm_encoder_v=use_bmm_encoder_v, use_triton_attention=use_triton_attention, optimizer=optimizer)
    _sync()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    for _ in range(steps):
        losses.append(float(_step(model, idx, targets, use_wide_encoder=use_wide_encoder, use_bmm_encoder_v=use_bmm_encoder_v, use_triton_attention=use_triton_attention, optimizer=optimizer)))
    _sync()
    elapsed = time.perf_counter() - started
    tokens = idx.numel() * steps
    return {
        "steps": steps,
        "tokens": tokens,
        "seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "last_loss": losses[-1],
        "finite_loss": all(torch.isfinite(torch.tensor(loss)) for loss in losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires real CUDA hardware")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    idx = torch.randint(args.vocab_size, (args.batch_size, args.sequence_length), device=device)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.sequence_length), device=device)

    config = BDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
        vocab_size=args.vocab_size, dropout=0.0,
    )

    def fresh_model() -> BDH:
        torch.manual_seed(args.seed)
        model = BDH(config).to(device=device, dtype=dtype)
        model.attn.freqs = model.attn.freqs.to(torch.float32)
        return model

    raw_model = fresh_model()

    def raw_step_fn():
        return raw_model(idx, targets)

    raw_optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.learning_rate, weight_decay=0.1, fused=True)
    for _ in range(args.warmup):
        raw_optimizer.zero_grad(set_to_none=True)
        _, loss = raw_model(idx, targets)
        loss.backward()
        raw_optimizer.step()
    _sync()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    raw_losses = []
    for _ in range(args.steps):
        raw_optimizer.zero_grad(set_to_none=True)
        _, loss = raw_model(idx, targets)
        loss.backward()
        raw_optimizer.step()
        raw_losses.append(float(loss.detach()))
    _sync()
    raw_elapsed = time.perf_counter() - started
    tokens = idx.numel() * args.steps
    raw_result = {
        "steps": args.steps, "tokens": tokens, "seconds": raw_elapsed,
        "tokens_per_second": tokens / raw_elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "last_loss": raw_losses[-1],
        "finite_loss": all(torch.isfinite(torch.tensor(loss)) for loss in raw_losses),
    }

    triton_model = fresh_model()
    triton_model.load_state_dict(raw_model.state_dict())
    triton_result = _benchmark(
        triton_model, idx, targets, use_wide_encoder=False, use_bmm_encoder_v=False, use_triton_attention=True,
        warmup=args.warmup, steps=args.steps, lr=args.learning_rate,
    )

    bmm_model = fresh_model()
    bmm_model.load_state_dict(raw_model.state_dict())
    bmm_result = _benchmark(
        bmm_model, idx, targets, use_wide_encoder=False, use_bmm_encoder_v=True, use_triton_attention=True,
        warmup=args.warmup, steps=args.steps, lr=args.learning_rate,
    )

    wide_model = fresh_model()
    wide_model.load_state_dict(raw_model.state_dict())
    wide_result = _benchmark(
        wide_model, idx, targets, use_wide_encoder=True, use_bmm_encoder_v=True, use_triton_attention=True,
        warmup=args.warmup, steps=args.steps, lr=args.learning_rate,
    )

    results = {
        "device": "cuda",
        "hardware_id": torch.cuda.get_device_name(device),
        "dtype": "bfloat16",
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "bdh_parameter_count": sum(p.numel() for p in raw_model.parameters()),
        "warmup_steps": args.warmup,
        "timed_steps": args.steps,
        "1_raw_oracle": raw_result,
        "2_triton_only": triton_result,
        "3_triton_plus_bmm_encoder_v": bmm_result,
        "4_triton_plus_bmm_plus_wide_encoder": wide_result,
        "2_over_1_speed_ratio": triton_result["tokens_per_second"] / raw_result["tokens_per_second"],
        "3_over_2_speed_ratio": bmm_result["tokens_per_second"] / triton_result["tokens_per_second"],
        "4_over_3_speed_ratio": wide_result["tokens_per_second"] / bmm_result["tokens_per_second"],
        "4_over_1_speed_ratio": wide_result["tokens_per_second"] / raw_result["tokens_per_second"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
