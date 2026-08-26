#!/usr/bin/env python3
"""Tier 2 item 15 of plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md:
a dedicated exact-SUPERSET coverage test, distinct from item 12's
exact-MATCH template-frequency metric. Item 12 asked "how many distinct
patterns are needed to exactly reproduce 90/95/99% of tokens' block
masks" (answer at production n/d=16, unreordered: 4-5 -- decent). This
script asks a different, more directly actionable question: is there a
single small FIXED candidate block set S (not per-token, one set for
everyone) such that virtually every token's true active-block set is a
SUBSET of S -- i.e. always including S's blocks and skipping everything
else would almost never miss a truly-active block?

S is built as the union of active blocks observed on a FIT sample, then
validated for real miss rate on a SEPARATE held-out EVAL sample (not the
sample S was built from) -- a real generalization test, not just fit
statistics. Also reports how |S| grows as more fit tokens are seen
(saturating vs. still-growing) since a superset that keeps growing
without bound isn't a usable fixed candidate set.
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


def superset_stats(fit_mask: torch.Tensor, eval_mask: torch.Tensor, block_size: int) -> dict:
    fit_blocks = blocks_active(fit_mask, block_size)
    eval_blocks = blocks_active(eval_mask, block_size)
    n_blocks = fit_blocks.shape[1]

    candidate_set = fit_blocks.any(dim=0)  # (n_blocks,) -- union of every active block seen in fit sample
    candidate_fraction = float(candidate_set.float().mean())

    # Saturation curve: union size after seeing 10/25/50/75/100% of fit tokens (prefix order, no reshuffling).
    checkpoints = [0.10, 0.25, 0.50, 0.75, 1.00]
    saturation = {}
    for frac in checkpoints:
        n_tok = max(1, int(round(fit_blocks.shape[0] * frac)))
        union_so_far = fit_blocks[:n_tok].any(dim=0)
        saturation[f"{int(frac*100)}pct_fit_tokens"] = float(union_so_far.float().mean())

    # Real held-out generalization: for each eval token, is its true active-block
    # set a SUBSET of the fit-derived candidate_set? (miss = a truly-active block
    # NOT in the candidate set -- would be silently skipped, a real error.)
    misses = eval_blocks & ~candidate_set.unsqueeze(0)  # (eval_tokens, n_blocks)
    tokens_with_any_miss = int((misses.any(dim=1)).sum())
    total_eval_tokens = eval_blocks.shape[0]

    return {
        "block_size": block_size,
        "n_blocks": n_blocks,
        "candidate_fraction": candidate_fraction,
        "saturation_by_fit_sample_size": saturation,
        "eval_tokens_with_at_least_one_miss": tokens_with_any_miss,
        "eval_tokens_with_at_least_one_miss_fraction": tokens_with_any_miss / total_eval_tokens,
        "total_eval_tokens": total_eval_tokens,
    }


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
    parser.add_argument("--block-sizes", type=str, default="16,32,64,128,256")
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

    block_sizes = [int(b) for b in args.block_sizes.split(",")]
    results = [superset_stats(fit_mask, eval_mask, bs) for bs in block_sizes]

    report = {
        "checkpoint": str(args.checkpoint),
        "n_total_neurons": fit_mask.shape[1],
        "fit_tokens": fit_mask.shape[0], "eval_tokens": eval_mask.shape[0],
        "results_by_block_size": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
