#!/usr/bin/env python3
"""Real quality check for BDH-Delta (reference/hz0h_bdh_delta_vnext_torch.py),
plans/newnewplan.md's "execute the entire plan" instruction, 2026-08-29.

Same methodology as every other Qwen/internal-computation sweep arm this
session: seed=7, hz0h_bytes_25m data, matched --target-tokens budget,
batch=8/seq=256, adamw, bfloat16, gradient checkpointing, decoder_up/
decoder_down SVD-warmstarted from the SAME checkpoint every other arm
uses (results/local/hz0h_bdh_checkpoint_for_ablation.pt) -- so val_loss
is directly comparable to the current champion (Phase 4 single-gate,
val_loss 1.4142/1.4326 seed=7 baseline depending on run, -0.0212/-0.0030
delta) and to every rejected arm (Muon +0.054, MTP +0.024, n-gram
+0.008, round-embed +0.0100, state-supervision +0.0504).

The one real methodology difference from every prior arm: curriculum
ramps n_refresh (K, the expensive exact-addressing count) instead of
n_layer, using the SAME curriculum_stages/depth_at helpers everyone else
uses, reusing config.n_layer as K's ramp target (0.5x, 0.75x, 1.0x of
n_refresh_final) -- n_think (M, the cheap axis) is NOT curriculum-ramped,
per the plan's own framing of K as the expensive axis.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_delta_vnext_torch import add_delta_vnext, bdh_delta_vnext_forward
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_combined_best_comparison import autocast_context, curriculum_stages, make_optimizer
from scripts.hz0h_bdh_vb_subspace_decoder_quality_check import svd_warmstart_decoder
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import depth_at, lr_at, read_batch


def train(config, args, device):
    torch.manual_seed(args.seed)
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.float32)
    if args.init_checkpoint is not None:
        svd_warmstart_decoder(model, args.init_checkpoint, config.subspace_rank, device)
    add_delta_vnext(model, n_refresh=args.n_refresh, n_think=args.n_think,
                     think_hidden=args.think_hidden, belief_hidden=args.belief_hidden,
                     alpha_init=args.alpha_init, beta_scale_init=args.beta_scale_init,
                     gamma_bias_init=args.gamma_bias_init, beta_bias_init=args.beta_bias_init)
    optimizer = make_optimizer(model.parameters(), args, device)
    steps = math.ceil(args.target_tokens / (args.batch_size * args.sequence_length))
    stages = curriculum_stages(args.target_tokens, args.n_refresh)  # ramps K, not n_layer
    tokens = 0
    started = time.perf_counter()
    with args.data.open() as handle:
        epochs = [0]
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, steps, args.warmup_steps, args.learning_rate)
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            n_refresh = depth_at(tokens, stages)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args, device):
                _, loss = bdh_delta_vnext_forward(model, idx, n_refresh, target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            tokens += args.batch_size * args.sequence_length
            if args.log_every and (step + 1) % args.log_every == 0:
                now = time.perf_counter()
                rate = tokens / (now - started)
                eta = (steps - step - 1) / max(step + 1, 1) * (now - started)
                print(f"[train_delta_vnext] step {step+1}/{steps} n_refresh={n_refresh} loss={float(loss):.4f} "
                      f"alpha={float(model.think_alpha):.4f} beta_scale={float(model.belief_beta_scale):.4f} "
                      f"{rate:.0f} tok/s eta={eta:.0f}s", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[train_delta_vnext] DONE {tokens} tokens in {elapsed:.0f}s final_loss={float(loss):.4f}", flush=True)
    model.eval()
    return model, elapsed


def evaluate_loss(model, args, device):
    epochs = [0]
    losses = []
    with args.validation_data.open() as handle, torch.no_grad(), autocast_context(args, device):
        for _ in range(args.eval_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            _, loss = bdh_delta_vnext_forward(model, idx, args.n_refresh, target)
            losses.append(float(loss))
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--init-checkpoint", type=Path, default=Path("results/local/hz0h_bdh_checkpoint_for_ablation.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-tokens", type=int, default=25_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8, help="unused by the delta forward itself, kept only "
                         "so BDHVBSubspaceDecoderConfig matches every other arm's config shape")
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--n-refresh", type=int, default=4, help="K, expensive exact-addressing count, curriculum-ramped")
    parser.add_argument("--n-think", type=int, default=2, help="M, cheap think-steps per refresh, NOT curriculum-ramped")
    parser.add_argument("--think-hidden", type=int, default=384)
    parser.add_argument("--belief-hidden", type=int, default=384)
    parser.add_argument("--alpha-init", type=float, default=0.5)
    parser.add_argument("--beta-scale-init", type=float, default=0.1)
    parser.add_argument("--gamma-bias-init", type=float, default=-4.0)
    parser.add_argument("--beta-bias-init", type=float, default=-5.0)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    config = BDHVBSubspaceDecoderConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
        d_state=args.d_state, subspace_rank=args.subspace_rank,
    )
    model, elapsed = train(config, args, device)
    val_loss = evaluate_loss(model, args, device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[delta_vnext] validation_loss={val_loss} params={params/1e6:.2f}M elapsed={elapsed:.0f}s", flush=True)
    print(f"[delta_vnext] final think_alpha={float(model.think_alpha):.4f} "
          f"belief_beta_scale={float(model.belief_beta_scale):.4f}", flush=True)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": vars(config), "state_dict": model.state_dict()}, args.save_checkpoint)
        print(f"[delta_vnext] saved checkpoint to {args.save_checkpoint}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "validation_loss": val_loss, "params": params, "elapsed_s": elapsed,
        "n_refresh": args.n_refresh, "n_think": args.n_think,
        "think_alpha": float(model.think_alpha), "belief_beta_scale": float(model.belief_beta_scale),
    }, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
