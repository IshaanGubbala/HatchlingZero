#!/usr/bin/env python3
"""Real first-pass quality-per-parameter check for the reasoning LoRA
adapter (reference/hz0h_bdh_hzcq_reasoning_lora_torch.py), never
previously exercised in an actual training run. Same warmstarted base
(results/local/hz0h_bdh_checkpoint_for_ablation.pt, SVD-reconstructed
decoder) as every other quality-check script this session, so the
frozen-base-only val_loss is directly comparable to the full-finetune
subspace-decoder baseline (1.7972 at 5M tokens). Base parameters are
FROZEN throughout -- only the LoRA A/B factors train -- to test the
real question: how much of the quality gap can a tiny number of extra
parameters close, relative to full fine-tuning.

Cheap local-first scale (small --target-tokens default) -- this is the
same "measure locally before any CUDA scale-up" discipline this
project has used for every architecture idea, not a final result.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_hzcq_reasoning_lora_torch import HZCQReasoningLoRA
from reference.hz0h_bdh_vb_subspace_decoder_checkpointed_torch import bdh_vb_subspace_decoder_forward_checkpointed
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_combined_best_comparison import curriculum_stages
from scripts.hz0h_bdh_vb_subspace_decoder_quality_check import svd_warmstart_decoder
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import depth_at, read_batch


def evaluate_loss(model, args, device):
    epochs = [0]
    losses = []
    with args.validation_data.open() as handle, torch.no_grad():
        for _ in range(args.eval_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            _, loss = bdh_vb_subspace_decoder_forward_checkpointed(model, idx, model.config.n_layer, target)
            losses.append(float(loss))
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--init-checkpoint", type=Path, default=Path("results/local/hz0h_bdh_checkpoint_for_ablation.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-tokens", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--no-curriculum", action="store_true",
                         help="Train at fixed full depth throughout instead of matching "
                              "hz0h_bdh_vb_subspace_decoder_quality_check.py's depth curriculum "
                              "(the old default behavior before this flag existed).")
    args = parser.parse_args()

    device = pick_device(args.device)
    config = BDHVBSubspaceDecoderConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
        d_state=args.d_state, subspace_rank=args.subspace_rank,
    )
    torch.manual_seed(args.seed)
    model = HZCQReasoningLoRA(config, rank=args.lora_rank, freeze_base=True).to(device=device, dtype=torch.float32)
    svd_warmstart_decoder(model, args.init_checkpoint, config.subspace_rank, device)

    adapter_params = model.adapter_parameter_count()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[lora_check] adapter_params={adapter_params} total_params={total_params/1e6:.2f}M "
          f"adapter_fraction={adapter_params/total_params:.5f}", flush=True)

    model.set_lora_scale(0.0)
    floor_loss = evaluate_loss(model, args, device)
    print(f"[lora_check] frozen-base-only (scale=0) val_loss={floor_loss:.4f}", flush=True)
    model.set_lora_scale(1.0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    steps = math.ceil(args.target_tokens / (args.batch_size * args.sequence_length))
    # Real, controlled-comparison fix: match hz0h_bdh_vb_subspace_decoder_quality_check.py's
    # own depth curriculum (curriculum_stages/depth_at) exactly, instead of always training
    # at fixed full depth -- the earlier 250K-token LoRA-vs-full-finetune head-to-head used
    # mismatched recipes (full-finetune ramped depth 4->8, LoRA always ran depth=n_layer),
    # so part of that gap could have been recipe, not parameter count. --no-curriculum
    # restores the old fixed-depth behavior for anyone who wants that instead.
    stages = curriculum_stages(args.target_tokens, config.n_layer) if not args.no_curriculum else [(args.target_tokens, config.n_layer)]
    tokens = 0
    started = time.perf_counter()
    epochs = [0]
    model.train()
    with args.data.open() as handle:
        for step in range(steps):
            lr_scale = min(1.0, (step + 1) / max(args.warmup_steps, 1))
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * lr_scale
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            depth = depth_at(tokens, stages)
            optimizer.zero_grad(set_to_none=True)
            _, loss = bdh_vb_subspace_decoder_forward_checkpointed(model, idx, depth, target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            tokens += args.batch_size * args.sequence_length
            if args.log_every and (step + 1) % args.log_every == 0:
                now = time.perf_counter()
                rate = tokens / (now - started)
                eta = (steps - step - 1) / max(step + 1, 1) * (now - started)
                print(f"[lora_check] step {step+1}/{steps} depth={depth} loss={float(loss):.4f} {rate:.0f} tok/s eta={eta:.0f}s", flush=True)

    synchronize(device)
    elapsed = time.perf_counter() - started
    model.eval()
    trained_loss = evaluate_loss(model, args, device)
    print(f"[lora_check] DONE {tokens} tokens in {elapsed:.0f}s trained_val_loss={trained_loss:.4f} "
          f"(floor was {floor_loss:.4f}, delta={floor_loss - trained_loss:+.4f})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "adapter_params": adapter_params, "total_params": total_params,
        "floor_val_loss_scale0": floor_loss, "trained_val_loss": trained_loss,
        "elapsed_seconds": elapsed, "tokens": tokens,
    }, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
