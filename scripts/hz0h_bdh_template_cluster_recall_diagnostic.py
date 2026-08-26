#!/usr/bin/env python3
"""Real test of the user's proposed template-codebook MoE idea (2026-08-25
conversation, following up on Tier 2 items 12/15): NOT a single static
candidate set (item 15's own test, already decisively negative at both
n/d=16 and n/d=128 -- 100% of blocks needed either way), but K=32-64
CLUSTERED templates, where each token routes to its nearest cluster and
that cluster's template (the union of its member tokens' active blocks,
computed on a FIT sample) becomes the candidate set actually computed.

This is the real per-token-adaptive version of "activation-template
analysis" the user asked for as experiment 1 of their own proposed
ranking. The stated gate: 32-64 templates should narrow the candidate
set to <=20-30% of blocks while retaining >99.9% recall, or the whole
direction isn't worth pursuing further.

Simple K-means-style clustering (no external dependency): centroids
initialized from random fit-sample tokens' own masks, a few rounds of
assign-by-L2-distance + recompute-centroid-as-mean. Each cluster's
TEMPLATE (the candidate set a router would actually use) is the UNION of
its FIT members' true active blocks -- guarantees exact recall for
however well the clustering + routing generalizes, not just fit
recall by construction (real held-out recall is what's reported).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig
from scripts.hz0h_bdh_neuron_reordering_diagnostic import collect_last_round_mask


def blocks_active(mask: torch.Tensor, block_size: int) -> torch.Tensor:
    n = mask.shape[1]
    n_blocks = n // block_size
    return mask.view(mask.shape[0], n_blocks, block_size).any(dim=-1)  # (tokens, n_blocks)


def kmeans_cluster(features: torch.Tensor, k: int, iterations: int, seed: int) -> torch.Tensor:
    """features: (tokens, n_blocks) float. Returns (tokens,) cluster assignment."""
    generator = torch.Generator().manual_seed(seed)
    n = features.shape[0]
    init_idx = torch.randperm(n, generator=generator)[:k]
    centroids = features[init_idx].clone()
    assignment = torch.zeros(n, dtype=torch.long)
    for _ in range(iterations):
        dists = torch.cdist(features, centroids)  # (tokens, k)
        assignment = dists.argmin(dim=1)
        for c in range(k):
            members = features[assignment == c]
            if members.shape[0] > 0:
                centroids[c] = members.mean(dim=0)
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--fit-batches", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--num-templates", type=str, default="32,64")
    parser.add_argument("--kmeans-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device)
    config = BDHConfig(**payload["config"])
    model = BDH(config).to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    epochs = [0]
    with args.validation_data.open() as handle:
        fit_mask = collect_last_round_mask(model, config, handle, args.fit_batches, args.batch_size, args.sequence_length, device, epochs)
        eval_mask = collect_last_round_mask(model, config, handle, args.eval_batches, args.batch_size, args.sequence_length, device, epochs)

    fit_blocks = blocks_active(fit_mask, args.block_size).float()  # (fit_tokens, n_blocks)
    eval_blocks = blocks_active(eval_mask, args.block_size)  # (eval_tokens, n_blocks) bool
    n_blocks = fit_blocks.shape[1]

    results = {}
    for K in (int(k) for k in args.num_templates.split(",")):
        assignment = kmeans_cluster(fit_blocks, K, args.kmeans_iterations, args.seed)

        templates = torch.zeros(K, n_blocks, dtype=torch.bool)
        for c in range(K):
            members = fit_blocks[assignment == c] > 0
            if members.shape[0] > 0:
                templates[c] = members.any(dim=0)  # UNION of this cluster's fit members' active blocks

        template_sizes = templates.float().mean(dim=1)  # (K,) candidate_fraction per template
        centroids = torch.stack([fit_blocks[assignment == c].mean(dim=0) if (assignment == c).any() else fit_blocks.mean(dim=0) for c in range(K)])

        eval_float = eval_blocks.float()
        eval_dists = torch.cdist(eval_float, centroids)  # (eval_tokens, K)
        eval_assignment = eval_dists.argmin(dim=1)

        eval_candidate_fraction = template_sizes[eval_assignment]  # (eval_tokens,)
        assigned_templates = templates[eval_assignment]  # (eval_tokens, n_blocks)
        misses = eval_blocks & ~assigned_templates
        tokens_with_miss = int(misses.any(dim=1).sum())
        recall_per_token = 1.0 - (misses.float().sum(dim=1) / eval_blocks.float().sum(dim=1).clamp(min=1))

        results[K] = {
            "num_templates": K,
            "template_candidate_fraction_mean": float(template_sizes.mean()),
            "template_candidate_fraction_max": float(template_sizes.max()),
            "template_candidate_fraction_min": float(template_sizes.min()),
            "eval_candidate_fraction_mean": float(eval_candidate_fraction.mean()),
            "eval_tokens_with_at_least_one_miss": tokens_with_miss,
            "eval_tokens_with_at_least_one_miss_fraction": tokens_with_miss / eval_blocks.shape[0],
            "eval_mean_recall": float(recall_per_token.mean()),
            "eval_min_recall": float(recall_per_token.min()),
        }
        print(f"K={K}: mean candidate_fraction={results[K]['eval_candidate_fraction_mean']:.4f} "
              f"| miss_rate={results[K]['eval_tokens_with_at_least_one_miss_fraction']:.4f} "
              f"| mean_recall={results[K]['eval_mean_recall']:.6f}", flush=True)

    report = {
        "checkpoint": str(args.checkpoint),
        "block_size": args.block_size, "n_blocks": n_blocks,
        "fit_tokens": fit_blocks.shape[0], "eval_tokens": eval_blocks.shape[0],
        "gate": "user's proposed gate: <=20-30% candidate fraction at >99.9% recall, else not worth pursuing",
        "results_by_num_templates": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
