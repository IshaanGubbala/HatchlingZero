#!/usr/bin/env python3
"""plans/newnewplan.md section 19 / section 28 priority 2: frequency
alone isn't the whole refresh-cadence story -- test WHERE the K=6
refreshes land, not just how many there are. Compares named placement
patterns (reference/hz0h_bdh_cached_evidence_torch.py's
NAMED_PLACEMENT_PATTERNS_8) against each other and against the
already-measured evenly-spaced 6/8 result (uniform_6, val_loss=1.4505
under the curriculum-ramped schedule -- see the K=6-constant rerun,
scripts/hz0h_bdh_cached_evidence_quality_check.py --constant-schedule,
for the like-for-like constant-schedule comparison point).

Deliberately uses a CONSTANT placement the entire run (no curriculum
ramp) -- the K=2 confound (curriculum_stages' max(2,...) floor
collapsing that arm's ramp to one stage while K=4/K=6 got real ramps)
already taught this project that ramp-shape and cadence/placement are
easy to conflate; this script avoids reintroducing that problem by
never ramping placement at all, so every pattern gets the exact same
training-schedule shape and only WHERE the refreshes land differs.

Same methodology otherwise as every other arm: seed=7, hz0h_bytes_25m
data, matched --target-tokens budget, batch=8/seq=256, adamw, bfloat16,
gradient checkpointing, decoder_up/decoder_down SVD-warmstarted, zero
new parameters beyond the existing g1 gate.
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

from reference.hz0h_bdh_cached_evidence_torch import NAMED_PLACEMENT_PATTERNS_8, bdh_cached_evidence_forward_checkpointed_explicit
from reference.hz0h_bdh_vb_subspace_decoder_gated_residual_torch import add_gated_residual_stream
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_combined_best_comparison import autocast_context, make_optimizer
from scripts.hz0h_bdh_vb_subspace_decoder_quality_check import svd_warmstart_decoder
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import lr_at, read_batch


def train(config, args, device, refresh_at):
    torch.manual_seed(args.seed)
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.float32)
    if args.init_checkpoint is not None:
        svd_warmstart_decoder(model, args.init_checkpoint, config.subspace_rank, device)
    add_gated_residual_stream(model, single_stream=True)
    optimizer = make_optimizer(model.parameters(), args, device)
    steps = math.ceil(args.target_tokens / (args.batch_size * args.sequence_length))
    tokens = 0
    started = time.perf_counter()
    with args.data.open() as handle:
        epochs = [0]
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, steps, args.warmup_steps, args.learning_rate)
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args, device):
                _, loss = bdh_cached_evidence_forward_checkpointed_explicit(model, idx, args.n_layer, refresh_at, target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            tokens += args.batch_size * args.sequence_length
            if args.log_every and (step + 1) % args.log_every == 0:
                now = time.perf_counter()
                rate = tokens / (now - started)
                eta = (steps - step - 1) / max(step + 1, 1) * (now - started)
                print(f"[train_placement] pattern={args.placement_pattern} step {step+1}/{steps} loss={float(loss):.4f} "
                      f"g1={float(model.g1):.4f} {rate:.0f} tok/s eta={eta:.0f}s", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[train_placement] DONE {tokens} tokens in {elapsed:.0f}s final_loss={float(loss):.4f}", flush=True)
    model.eval()
    return model, elapsed


def evaluate_loss(model, args, device, refresh_at):
    epochs = [0]
    losses = []
    with args.validation_data.open() as handle, torch.no_grad(), autocast_context(args, device):
        for _ in range(args.eval_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            _, loss = bdh_cached_evidence_forward_checkpointed_explicit(model, idx, args.n_layer, refresh_at, target)
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
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--placement-pattern", choices=sorted(NAMED_PLACEMENT_PATTERNS_8), required=True)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    refresh_at = NAMED_PLACEMENT_PATTERNS_8[args.placement_pattern]
    device = pick_device(args.device)
    config = BDHVBSubspaceDecoderConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
        d_state=args.d_state, subspace_rank=args.subspace_rank,
    )
    model, elapsed = train(config, args, device, refresh_at)
    val_loss = evaluate_loss(model, args, device, refresh_at)
    params = sum(p.numel() for p in model.parameters())
    print(f"[placement] pattern={args.placement_pattern} refresh_at={sorted(refresh_at)} "
          f"validation_loss={val_loss} params={params/1e6:.2f}M elapsed={elapsed:.0f}s "
          f"final_g1={float(model.g1):.4f}", flush=True)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": vars(config), "state_dict": model.state_dict()}, args.save_checkpoint)
        print(f"[placement] saved checkpoint to {args.save_checkpoint}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "validation_loss": val_loss, "params": params, "elapsed_s": elapsed,
        "n_layer": args.n_layer, "placement_pattern": args.placement_pattern,
        "refresh_at": sorted(refresh_at), "n_refresh": len(refresh_at), "final_g1": float(model.g1),
    }, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
