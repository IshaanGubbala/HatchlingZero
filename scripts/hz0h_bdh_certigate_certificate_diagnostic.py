#!/usr/bin/env python3
"""Tier 2 item 14 of plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md
(section 10, Phase C-A): can whole blocks of the encoder projection be
PROVABLY skipped -- zero approximation, not a learned predictor -- before
computing them at all?

For a block g of K encoder-output columns e_gj (j=1..K), with centroid
c_g = mean_j(e_gj) and radius rho_g = max_j ||e_gj - c_g||_2, Cauchy-
Schwarz gives, for any input x:

    x . e_gj = x . c_g + x . (e_gj - c_g) <= x . c_g + ||x||_2 * rho_g

So if x . c_g + ||x||_2 * rho_g < 0, then x . e_gj < 0 for EVERY j in the
block -- every neuron in the block is guaranteed below the ReLU
threshold, and the entire block can be skipped with exactly zero
approximation error. This is a real inequality, not a heuristic: the
false-negative rate (certified-off blocks that were actually active)
MUST be exactly zero if the implementation is correct -- this script
verifies that directly against the real ReLU ground truth, not just
reports a rate and hopes it's low.

Scoped to the LAST recurrent round only (steady-state activity), same
scoping choice as scripts/hz0h_bdh_filterability_diagnostic.py and
scripts/hz0h_bdh_neuron_reordering_diagnostic.py.
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
from scripts.hz0h_factorized_curriculum_full_comparison import read_batch


def certificate_stats(x_inputs: torch.Tensor, encoder_weight: torch.Tensor, block_size: int) -> dict:
    """x_inputs: (num_tokens, D). encoder_weight: (nh, D, N). Returns real
    certified-off statistics for this block_size, verified against ground
    truth ReLU activity computed from the SAME weight and inputs."""
    nh, D, N = encoder_weight.shape
    assert N % block_size == 0, f"N={N} not divisible by block_size={block_size}"
    n_blocks = N // block_size

    blocks = encoder_weight.view(nh, D, n_blocks, block_size)  # (nh, D, n_blocks, block_size)
    centroid = blocks.mean(dim=-1)  # (nh, D, n_blocks)
    delta = blocks - centroid.unsqueeze(-1)  # (nh, D, n_blocks, block_size)
    delta_norms = delta.norm(dim=1)  # (nh, n_blocks, block_size) -- per-column deviation norm
    rho = delta_norms.max(dim=-1).values  # (nh, n_blocks) -- block radius

    x_norm = x_inputs.norm(dim=-1)  # (num_tokens,)
    # score[t, h, g] = x_t . c_{h,g} + ||x_t|| * rho_{h,g}
    dot = torch.einsum("td,hdg->thg", x_inputs, centroid)  # (tokens, nh, n_blocks)
    score = dot + x_norm.view(-1, 1, 1) * rho.view(1, nh, n_blocks)
    certified_off = score < 0  # (tokens, nh, n_blocks)

    # Ground truth: actually compute the encoder output and check whether every
    # neuron in each block is exactly non-positive (ReLU output is exactly zero).
    raw = torch.einsum("td,hdn->thn", x_inputs, encoder_weight)  # (tokens, nh, N)
    raw_blocks = raw.view(raw.shape[0], nh, n_blocks, block_size)
    actually_off = (raw_blocks <= 0).all(dim=-1)  # (tokens, nh, n_blocks)

    false_negatives = certified_off & ~actually_off  # certified off but was NOT actually off -- should be zero
    false_negative_count = int(false_negatives.sum())

    return {
        "block_size": block_size,
        "n_blocks_per_head": n_blocks,
        "fraction_certified_off": float(certified_off.float().mean()),
        "candidate_fraction": float((~certified_off).float().mean()),
        "false_negative_count": false_negative_count,
        "false_negative_rate": false_negative_count / certified_off.numel(),
        "total_block_evaluations": certified_off.numel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--block-sizes", type=str, default="16,32,64,128,256")
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device)
    config = BDHConfig(**payload["config"])
    model = BDH(config).to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    last_round = config.n_layer - 1
    all_x_inputs = []
    epochs = [0]
    with args.validation_data.open() as handle, torch.no_grad():
        for _ in range(args.eval_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx = data[:, :-1].contiguous()
            B, T = idx.shape
            x = model.ln(model.embed(idx).unsqueeze(1))
            for level in range(config.n_layer):
                x_latent = x @ model._w(model.encoder)
                x_sparse = F.relu(x_latent)
                if level == last_round:
                    all_x_inputs.append(x.squeeze(1).reshape(B * T, -1))  # (B*T, D) -- the actual encoder INPUT this round
                yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
                yKV = model.ln(yKV)
                y_latent = yKV @ model._w(model.encoder_v)
                y_sparse = F.relu(y_latent)
                g = x_sparse * y_sparse
                xy_sparse = model.drop(g)
                yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, config.n_embd * config.mlp_internal_dim_multiplier // config.n_head * config.n_head) @ model._w(model.decoder)
                y = model.ln(yMLP)
                x = model.ln(x + y)

    x_inputs = torch.cat(all_x_inputs, dim=0)
    encoder_weight = model._w(model.encoder)  # (nh, D, N)

    block_sizes = [int(b) for b in args.block_sizes.split(",")]
    with torch.no_grad():
        results = [certificate_stats(x_inputs, encoder_weight, bs) for bs in block_sizes]

    total_fn = sum(r["false_negative_count"] for r in results)
    report = {
        "checkpoint": str(args.checkpoint),
        "scope_note": "last recurrent round only, real trained encoder weight, real activation inputs from held-out validation data.",
        "n_head": config.n_head, "n_embd": config.n_embd, "N_per_head": encoder_weight.shape[-1],
        "total_tokens": x_inputs.shape[0],
        "results_by_block_size": results,
        "total_false_negatives_across_all_block_sizes": total_fn,
        "correctness_note": "false_negative_count MUST be exactly 0 for every block size if the certificate math/implementation is correct -- a real bug, not just a bad rate, if nonzero.",
        "gate": ">=50% of blocks certified dead OR <=25-30% candidate neuron fraction, at block sizes large enough for efficient dense GEMMs, is the target per the plan's own Phase C-A gate.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
