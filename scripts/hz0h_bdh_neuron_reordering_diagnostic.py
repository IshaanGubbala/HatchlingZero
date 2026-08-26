#!/usr/bin/env python3
"""Tier 2 item 13 of plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md
(Phase BORG-A, section 11): can a single, consistent, offline neuron
permutation turn BDH's arbitrary-looking exact activation sparsity into
useful BLOCK sparsity? A pure coordinate relabeling changes nothing about
model quality (same math, different neuron indices) -- only whether the
GPU sees contiguous skippable blocks instead of scattered singletons.

Method (spectral seriation, no scipy dependency): build an n x n neuron
co-activation affinity matrix (how often each pair of neurons fires
together) on a FIT sample, take its leading eigenvector via torch.linalg.
eigh, and sort neurons by that eigenvector's value -- a standard,
principled ordering technique that places frequently-co-active neurons
near each other. The permutation is then measured on a SEPARATE eval
sample (not the fit sample) to avoid trivially overfitting the ordering
to the exact data being scored.

Reuses block_occupancy/template_coverage from
scripts/hz0h_bdh_filterability_diagnostic.py unmodified -- this script
only adds the permutation-fitting step, then calls the same measurement
functions before and after reordering for a real before/after
comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig
from scripts.hz0h_bdh_filterability_diagnostic import block_occupancy, template_coverage
from scripts.hz0h_factorized_curriculum_full_comparison import read_batch


def collect_last_round_mask(model: BDH, config: BDHConfig, handle, batches: int, batch_size: int, seq_len: int, device, epochs) -> torch.Tensor:
    nh = config.n_head
    D = config.n_embd
    N = D * config.mlp_internal_dim_multiplier // nh
    last_round = config.n_layer - 1
    masks = []
    with torch.no_grad():
        for _ in range(batches):
            data = read_batch(handle, batch_size, seq_len, device, epochs)
            idx = data[:, :-1].contiguous()
            B, T = idx.shape
            x = model.ln(model.embed(idx).unsqueeze(1))
            for level in range(config.n_layer):
                x_latent = x @ model._w(model.encoder)
                x_sparse = F.relu(x_latent)
                if level == last_round:
                    mask = (x_sparse != 0).permute(0, 2, 1, 3).reshape(B, T, nh * N)
                    for b in range(B):
                        masks.append(mask[b])
                yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
                yKV = model.ln(yKV)
                y_latent = yKV @ model._w(model.encoder_v)
                y_sparse = F.relu(y_latent)
                g = x_sparse * y_sparse
                xy_sparse = model.drop(g)
                yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
                y = model.ln(yMLP)
                x = model.ln(x + y)
    return torch.cat(masks, dim=0)


def spectral_seriation_order(fit_mask: torch.Tensor) -> torch.Tensor:
    """fit_mask: (tokens, n_neurons) bool. Returns a permutation (n_neurons,)
    ordering neurons by the leading eigenvector of their co-activation
    affinity matrix -- a standard seriation technique to reveal block
    structure in a co-occurrence matrix without needing scipy."""
    m = fit_mask.float()
    affinity = m.T @ m  # (n, n) co-occurrence counts, symmetric, PSD
    eigenvalues, eigenvectors = torch.linalg.eigh(affinity)
    leading = eigenvectors[:, -1]  # eigh returns ascending eigenvalues
    return torch.argsort(leading)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--fit-batches", type=int, default=8, help="Batches used to COMPUTE the permutation.")
    parser.add_argument("--eval-batches", type=int, default=8, help="SEPARATE batches used to MEASURE occupancy before/after -- not the same data the permutation was fit on.")
    parser.add_argument("--block-sizes", type=str, default="16,32,64,128,256")
    parser.add_argument("--template-block-size", type=int, default=64)
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

    perm = spectral_seriation_order(fit_mask)
    eval_mask_reordered = eval_mask[:, perm]

    block_sizes = [int(x) for x in args.block_sizes.split(",")]

    before = {
        "block_occupancy": [block_occupancy(eval_mask, bs) for bs in block_sizes],
        "template_coverage": template_coverage(eval_mask, args.template_block_size, [0.90, 0.95, 0.99]),
    }
    after = {
        "block_occupancy": [block_occupancy(eval_mask_reordered, bs) for bs in block_sizes],
        "template_coverage": template_coverage(eval_mask_reordered, args.template_block_size, [0.90, 0.95, 0.99]),
    }

    report = {
        "checkpoint": str(args.checkpoint),
        "method": "spectral seriation: leading eigenvector of the neuron co-activation affinity matrix, "
                  "fit on a SEPARATE sample from the one occupancy is measured on (avoids overfitting the "
                  "permutation to the exact evaluation data).",
        "n_total_neurons": fit_mask.shape[1],
        "fit_tokens": fit_mask.shape[0],
        "eval_tokens": eval_mask.shape[0],
        "before_reorder": before,
        "after_reorder": after,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
