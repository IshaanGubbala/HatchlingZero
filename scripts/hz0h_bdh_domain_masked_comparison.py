#!/usr/bin/env python3
"""First real signal check for the structural domain-specialization
design (2026-08-21): does hard domain-block masking
(`reference/hz0h_bdh_domain_masked_torch.py`, proven to give an EXACT
gradient firewall per block) actually change per-domain quality versus
an ordinary dense model, at local scale, before committing to any real
CUDA spend? Trains TWO arms on the same real token budget and domain
data (`scripts/hz0h_bdh_domain_bytes_prep.py`'s per-domain train/val
splits):

- `dense`: ordinary `bdh_variable_depth_forward`, trained on the
  round-robin-mixed stream (same as Part 11's domain-specialization
  diagnostic used for its own dense baseline).
- `hard_routed`: `bdh_domain_masked_forward`, each training step picks
  one domain uniformly at random, reads a batch from THAT domain's own
  train split, and masks to shared+that domain's block only -- matching
  the real per-batch routing the proposal describes ("During a math
  batch: shared and math neurons execute... only shared and math
  parameters receive gradients").

Evaluates BOTH arms on every domain's own held-out validation split
(dense with no mask -- it never learned to route; hard_routed with each
domain's own mask), reporting real per-domain validation loss side by
side. Real, deliberate scope for this first check: no auxiliary losses
(cross-domain suppression, orthogonality, capacity balancing, expert
dropout, mixed-domain blending) and no learned/soft router arm yet --
those are real, disclosed follow-up work from the full 2026-08-21
proposal, held back until this two-arm check shows the core mechanism
does something worth building on.

Real, disclosed limits: local-scale only (small width/token budget) --
a first-pass signal check, matching this project's own established
pattern (Part 5, Part 6, Part 11's own local-vs-CUDA gap) of a cheap
local prototype before a real CUDA-scale confirmation is worth
dispatching.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_domain_masked_torch import bdh_domain_masked_forward, build_domain_mask, domain_block_layout
from reference.hz0h_bdh_torch import BDHConfig
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward
from scripts.hz0h_bdh_combined_best_comparison import autocast_context, make_optimizer
from scripts.hz0h_bdh_domain_bytes_prep import DOMAINS
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import lr_at, read_batch


def train_dense(config: BDHConfig, args, device) -> tuple:
    torch.manual_seed(args.seed)
    from reference.hz0h_bdh_torch import BDH
    model = BDH(config).to(device=device, dtype=torch.float32)
    optimizer = make_optimizer(model.parameters(), args, device)
    steps = math.ceil(args.target_tokens / (args.batch_size * args.sequence_length))
    epochs = [0]
    started = time.perf_counter()
    with (args.domains_dir / "mixed_train.jsonl").open() as handle:
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, steps, args.warmup_steps, args.learning_rate)
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args, device):
                _, loss = bdh_variable_depth_forward(model, idx, config.n_layer, target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if args.log_every and (step + 1) % args.log_every == 0:
                print(f"[dense] step {step+1}/{steps} loss={float(loss):.4f}", flush=True)
    synchronize(device)
    print(f"[dense] DONE in {time.perf_counter()-started:.0f}s final_loss={float(loss):.4f}", flush=True)
    model.eval()
    return model, None


def train_hard_routed(config: BDHConfig, args, device, layout: dict, N: int) -> tuple:
    torch.manual_seed(args.seed)
    from reference.hz0h_bdh_torch import BDH
    model = BDH(config).to(device=device, dtype=torch.float32)
    optimizer = make_optimizer(model.parameters(), args, device)
    steps = math.ceil(args.target_tokens / (args.batch_size * args.sequence_length))

    handles = {name: (args.domains_dir / f"{name}_train.jsonl").open() for name in DOMAINS}
    epochs = {name: [0] for name in DOMAINS}
    masks = {name: build_domain_mask(layout, name, N, device=device) for name in DOMAINS}

    started = time.perf_counter()
    domain_counts = {name: 0 for name in DOMAINS}
    try:
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, steps, args.warmup_steps, args.learning_rate)
            domain = random.choice(DOMAINS)
            domain_counts[domain] += 1
            data = read_batch(handles[domain], args.batch_size, args.sequence_length, device, epochs[domain])
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args, device):
                _, loss = bdh_domain_masked_forward(model, idx, config.n_layer, masks[domain], target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if args.log_every and (step + 1) % args.log_every == 0:
                print(f"[hard_routed] step {step+1}/{steps} domain={domain} loss={float(loss):.4f}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    synchronize(device)
    print(f"[hard_routed] DONE in {time.perf_counter()-started:.0f}s final_loss={float(loss):.4f} "
          f"domain_counts={domain_counts}", flush=True)
    model.eval()
    return model, masks


def evaluate_per_domain(model, args, device, n_layer: int, masks: dict | None) -> dict:
    results = {}
    for name in DOMAINS:
        epochs = [0]
        losses = []
        mask = masks[name] if masks is not None else None
        with (args.domains_dir / f"{name}_val.jsonl").open() as handle, torch.no_grad(), autocast_context(args, device):
            for _ in range(args.eval_batches):
                data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
                idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
                if mask is not None:
                    _, loss = bdh_domain_masked_forward(model, idx, n_layer, mask, target)
                else:
                    _, loss = bdh_variable_depth_forward(model, idx, n_layer, target)
                losses.append(float(loss))
        results[name] = sum(losses) / len(losses)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains-dir", type=Path, default=Path("data/packed/domains"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-tokens", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--shared-fraction", type=float, default=0.25)
    parser.add_argument("--eval-batches", type=int, default=15)
    args = parser.parse_args()

    device = pick_device(args.device)
    random.seed(args.seed)

    config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0)
    N = args.n_embd * args.mult // args.n_head
    layout = domain_block_layout(N, DOMAINS, shared_fraction=args.shared_fraction)
    print(f"=== layout: N={N}, shared={layout['shared'].numel()}, "
          f"per-domain={[layout[d].numel() for d in DOMAINS]} ===", flush=True)

    print("=== training dense arm ===", flush=True)
    dense_model, _ = train_dense(config, args, device)
    dense_per_domain = evaluate_per_domain(dense_model, args, device, args.n_layer, None)
    print(f"[dense] per-domain validation_loss: {dense_per_domain}", flush=True)
    del dense_model

    print("=== training hard_routed arm ===", flush=True)
    hard_model, masks = train_hard_routed(config, args, device, layout, N)
    hard_per_domain = evaluate_per_domain(hard_model, args, device, args.n_layer, masks)
    print(f"[hard_routed] per-domain validation_loss: {hard_per_domain}", flush=True)

    report = {
        "config": {"n_embd": args.n_embd, "mult": args.mult, "n_layer": args.n_layer, "n_head": args.n_head,
                    "N_per_head": N, "shared_fraction": args.shared_fraction, "target_tokens": args.target_tokens,
                    "seed": args.seed},
        "layout": {name: idx.tolist() for name, idx in layout.items()},
        "dense_per_domain_validation_loss": dense_per_domain,
        "hard_routed_per_domain_validation_loss": hard_per_domain,
        "dense_mean": sum(dense_per_domain.values()) / len(dense_per_domain),
        "hard_routed_mean": sum(hard_per_domain.values()) / len(hard_per_domain),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
