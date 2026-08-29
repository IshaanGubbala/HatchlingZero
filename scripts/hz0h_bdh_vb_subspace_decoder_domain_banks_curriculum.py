#!/usr/bin/env python3
"""Real test of "Developmental Neuron Specialization" (proposed
2026-08-28, refined into a write-side-only, fixed-domain-lookup,
hard-freeze design): does giving the compound model a domain-specific
write bank per real domain, activated ONLY during that domain's
SEQUENTIAL training phase (not i.i.d.-mixed), reduce cross-domain
forgetting relative to a dense model trained on the identical
sequential curriculum?

Real prior evidence this test is built on top of, not ignoring:
`scripts/hz0h_bdh_domain_specialization_diagnostic.py` already found,
at production scale with CUDA confirmation, that i.i.d.-mixed domain
training produces ~zero addressing-neuron domain specialization
(within/across Jaccard ratio 1.03x-1.12x, essentially flat). This test
uses a SEQUENTIAL curriculum instead (a genuinely different regime,
closer to catastrophic-forgetting setups) and partitions the WRITE
side, not addressing -- so the headline measurement here is per-domain
validation loss (does banking resist forgetting), with the
addressing-neuron within/across Jaccard logged as a secondary,
lower-expected-yield measurement given the prior result.

Five real domains, five real sequential phases, `target-tokens-per-domain`
each (default 5M, matching the real per-domain corpus sizes -- 10.4M-
15.2M bytes each, comfortably above 5M tokens with no forced repetition
for either arm's own read-through).
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

from reference.hz0h_bdh_vb_subspace_decoder_checkpointed_torch import bdh_vb_subspace_decoder_forward_checkpointed
from reference.hz0h_bdh_vb_subspace_decoder_domain_banks_torch import (
    add_domain_banks,
    bdh_vb_subspace_decoder_forward_dense_with_support_checkpointed,
    bdh_vb_subspace_decoder_forward_domain_banks_checkpointed,
)
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_combined_best_comparison import autocast_context, curriculum_stages
from scripts.hz0h_bdh_g_r_operator_diagnostic import cross_domain_support_jaccard, cross_token_support_jaccard
from scripts.hz0h_bdh_vb_subspace_decoder_quality_check import svd_warmstart_decoder
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import depth_at, lr_at, read_batch

DOMAIN_ORDER = ["code", "documentation", "json_and_configuration", "mathematical_and_structured", "terminal_and_debugging"]


def train_curriculum(config, args, device, use_banks: bool):
    torch.manual_seed(args.seed)
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.float32)
    if args.init_checkpoint is not None:
        svd_warmstart_decoder(model, args.init_checkpoint, config.subspace_rank, device)
    if use_banks:
        add_domain_banks(model, n_domains=len(DOMAIN_ORDER))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))

    steps_per_domain = math.ceil(args.target_tokens_per_domain / (args.batch_size * args.sequence_length))
    total_steps = steps_per_domain * len(DOMAIN_ORDER)
    total_tokens_target = args.target_tokens_per_domain * len(DOMAIN_ORDER)
    stages = curriculum_stages(total_tokens_target, config.n_layer)

    tokens = 0
    global_step = 0
    started = time.perf_counter()
    for domain_id, domain_name in enumerate(DOMAIN_ORDER):
        data_path = args.domains_dir / f"{domain_name}_train.jsonl"
        epochs = [0]
        with data_path.open() as handle:
            for _local_step in range(steps_per_domain):
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(global_step, total_steps, args.warmup_steps, args.learning_rate)
                data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
                idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
                depth = depth_at(tokens, stages)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(args, device):
                    if use_banks:
                        _, loss = bdh_vb_subspace_decoder_forward_domain_banks_checkpointed(model, idx, depth, domain_id, target)
                    else:
                        _, loss = bdh_vb_subspace_decoder_forward_checkpointed(model, idx, depth, target)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                tokens += args.batch_size * args.sequence_length
                global_step += 1
                if args.log_every and global_step % args.log_every == 0:
                    now = time.perf_counter()
                    rate = tokens / (now - started)
                    print(f"[curriculum:{'banks' if use_banks else 'dense'}] domain={domain_name} "
                          f"step {global_step}/{total_steps} depth={depth} loss={float(loss):.4f} {rate:.0f} tok/s", flush=True)
        print(f"[curriculum:{'banks' if use_banks else 'dense'}] finished domain={domain_name} "
              f"({tokens} total tokens so far)", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    model.eval()
    return model, elapsed


def evaluate_all_domains(model, args, device, use_banks: bool) -> dict:
    results = {}
    for domain_id, domain_name in enumerate(DOMAIN_ORDER):
        val_path = args.domains_dir / f"{domain_name}_val.jsonl"
        epochs = [0]
        losses = []
        with val_path.open() as handle, torch.no_grad(), autocast_context(args, device):
            for _ in range(args.eval_batches):
                data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
                idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
                if use_banks:
                    _, loss = bdh_vb_subspace_decoder_forward_domain_banks_checkpointed(model, idx, model.config.n_layer, domain_id, target)
                else:
                    _, loss = bdh_vb_subspace_decoder_forward_checkpointed(model, idx, model.config.n_layer, target)
                losses.append(float(loss))
        results[domain_name] = sum(losses) / len(losses)
    return results


def collect_support_reservoirs(model, args, device, use_banks: bool, N: int) -> dict:
    reservoirs = {}
    for domain_id, domain_name in enumerate(DOMAIN_ORDER):
        val_path = args.domains_dir / f"{domain_name}_val.jsonl"
        epochs = [0]
        samples = []
        generator = torch.Generator().manual_seed(args.seed)
        with val_path.open() as handle, torch.no_grad(), autocast_context(args, device):
            for _ in range(args.support_eval_batches):
                data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
                idx = data[:, :-1].contiguous()
                if use_banks:
                    _, _, support = bdh_vb_subspace_decoder_forward_domain_banks_checkpointed(
                        model, idx, model.config.n_layer, domain_id, targets=None, collect_support=True)
                else:
                    _, _, support = bdh_vb_subspace_decoder_forward_dense_with_support_checkpointed(
                        model, idx, model.config.n_layer, targets=None, collect_support=True)
                flat = support.reshape(-1, N).float()
                n = flat.shape[0]
                take = min(args.support_reservoir_per_batch, n)
                keep = torch.randperm(n, generator=generator)[:take]
                samples.append(flat[keep].cpu())
        reservoirs[domain_name] = torch.cat(samples, dim=0)
    return reservoirs


def jaccard_summary(reservoirs: dict) -> dict:
    within = []
    for name, res in reservoirs.items():
        within.append(cross_token_support_jaccard(res)["mean_jaccard"])
    across = []
    names = list(reservoirs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            across.append(cross_domain_support_jaccard(reservoirs[names[i]], reservoirs[names[j]])["mean_jaccard"])
    mean_within = sum(within) / len(within)
    mean_across = sum(across) / len(across)
    return {"mean_within": mean_within, "mean_across": mean_across, "ratio": mean_within / mean_across if mean_across > 0 else float("nan"),
            "per_domain_within": dict(zip(names, within))}


def run_arm(config, args, device, use_banks: bool) -> dict:
    label = "banks" if use_banks else "dense"
    print(f"=== training {label} arm ===", flush=True)
    model, elapsed = train_curriculum(config, args, device, use_banks)
    val_losses = evaluate_all_domains(model, args, device, use_banks)
    print(f"[{label}] per-domain val_loss: {val_losses}", flush=True)
    N = config.n_embd * config.mlp_internal_dim_multiplier // config.n_head
    reservoirs = collect_support_reservoirs(model, args, device, use_banks, N)
    jaccard = jaccard_summary(reservoirs)
    print(f"[{label}] jaccard within/across ratio: {jaccard['ratio']:.4f} "
          f"(within={jaccard['mean_within']:.4f} across={jaccard['mean_across']:.4f})", flush=True)
    params = sum(p.numel() for p in model.parameters())
    return {"val_losses_by_domain": val_losses, "jaccard": jaccard, "training_seconds": elapsed, "parameter_count": params}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains-dir", type=Path, default=Path("data/packed/domains"))
    parser.add_argument("--init-checkpoint", type=Path, default=Path("results/local/hz0h_bdh_checkpoint_for_ablation.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-tokens-per-domain", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--support-eval-batches", type=int, default=15)
    parser.add_argument("--support-reservoir-per-batch", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--arm", choices=["dense", "banks", "both"], default="both")
    args = parser.parse_args()

    device = pick_device(args.device)
    config = BDHVBSubspaceDecoderConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
        d_state=args.d_state, subspace_rank=args.subspace_rank,
    )

    report = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "domain_order": DOMAIN_ORDER}
    if args.arm in ("dense", "both"):
        report["dense"] = run_arm(config, args, device, use_banks=False)
    if args.arm in ("banks", "both"):
        report["banks"] = run_arm(config, args, device, use_banks=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
