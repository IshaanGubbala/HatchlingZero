#!/usr/bin/env python3
"""Tier 2 item 12 of plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md
(Phase S-C, "measure filterability, not just sparsity"): for a trained
checkpoint, characterize whether its x-activation mask (the mask that
controls exact skip opportunities -- see the plan's own note that
x_density, not g_density, gates E_v/decoder skipping) can be organized
into hardware-friendly BLOCK structure, not just raw sparsity percentage.

Computed on the LAST recurrent round only (steady-state activity), not
averaged across all rounds -- a real scoping choice, disclosed here
rather than silently narrowing scope: the plan's own per-round density
numbers (scripts/hz0h_bdh_x_sparsity_diagnostic.py) already cover the
round-by-round picture; this script adds the metrics that measurement
never touched (block occupancy, run-lengths, cross-token Jaccard, mask
entropy, template coverage, static-top-K candidate recall).

Explicitly NOT computed here: "certifiable-off fraction under block
bounds" (the plan's own §10 Phase C-A CertiGate diagnostic) -- that
needs the encoder weight matrix's per-block radius/centroid structure
(rho_g, c_g), not just activation statistics, and belongs to a later,
separate phase per the plan's own ordering (S-C now, C-A only "after
exact downstream skipping is validated").
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


def block_occupancy(mask: torch.Tensor, block_size: int) -> dict:
    """mask: (num_tokens, n_neurons) bool. A block is 'active' if ANY
    neuron in it is active for that token."""
    n = mask.shape[1]
    assert n % block_size == 0, f"n_neurons={n} not divisible by block_size={block_size}"
    n_blocks = n // block_size
    blocks = mask.view(mask.shape[0], n_blocks, block_size).any(dim=-1)  # (tokens, n_blocks)
    occupancy_fraction = float(blocks.float().mean())
    active_blocks_per_token = blocks.sum(dim=-1).float()
    return {
        "block_size": block_size,
        "n_blocks": n_blocks,
        "occupancy_fraction": occupancy_fraction,
        "active_blocks_per_token_mean": float(active_blocks_per_token.mean()),
        "active_blocks_per_token_std": float(active_blocks_per_token.std()),
    }


def run_length_stats(mask: torch.Tensor) -> dict:
    """Contiguous-run lengths of True values along the neuron dimension,
    pooled across all tokens in this batch of masks."""
    lengths: list[int] = []
    mask_np = mask.cpu()
    for row in mask_np:
        run = 0
        for val in row.tolist():
            if val:
                run += 1
            elif run > 0:
                lengths.append(run)
                run = 0
        if run > 0:
            lengths.append(run)
    if not lengths:
        return {"mean": 0.0, "median": 0.0, "max": 0, "count": 0}
    lengths_t = torch.tensor(lengths, dtype=torch.float32)
    return {
        "mean": float(lengths_t.mean()),
        "median": float(lengths_t.median()),
        "max": int(lengths_t.max()),
        "count": len(lengths),
    }


def nearby_token_jaccard(mask: torch.Tensor, distances: list[int]) -> dict:
    """mask: (T, n_neurons) bool for a SINGLE sequence. Jaccard(active(t),
    active(t+d)) averaged over valid t, per distance."""
    T = mask.shape[0]
    out = {}
    for d in distances:
        if d >= T:
            out[str(d)] = None
            continue
        a = mask[: T - d]
        b = mask[d:]
        intersection = (a & b).sum(dim=-1).float()
        union = (a | b).sum(dim=-1).float()
        valid = union > 0
        if not bool(valid.any()):
            out[str(d)] = None
            continue
        jaccard = (intersection[valid] / union[valid]).mean()
        out[str(d)] = float(jaccard)
    return out


def mask_entropy(mask: torch.Tensor) -> float:
    """Mean per-neuron marginal Bernoulli entropy (bits) across the
    batch of masks -- how informative/predictable each neuron's
    activation is in isolation, not the joint pattern entropy."""
    p = mask.float().mean(dim=0).clamp(1e-6, 1 - 1e-6)
    entropy = -(p * p.log2() + (1 - p) * (1 - p).log2())
    return float(entropy.mean())


def template_coverage(mask: torch.Tensor, block_size: int, coverage_targets: list[float]) -> dict:
    """Quantize each token's mask into a per-block active/inactive
    pattern (a 'template'), count exact-pattern frequency, report how
    many of the most-common templates are needed to exact-cover the
    given coverage fractions of all tokens."""
    n = mask.shape[1]
    n_blocks = n // block_size
    blocks = mask.view(mask.shape[0], n_blocks, block_size).any(dim=-1)  # (tokens, n_blocks)
    packed = [tuple(row.tolist()) for row in blocks]
    counts: dict[tuple, int] = {}
    for pattern in packed:
        counts[pattern] = counts.get(pattern, 0) + 1
    total = len(packed)
    sorted_counts = sorted(counts.values(), reverse=True)
    result = {"block_size": block_size, "total_tokens": total, "unique_templates": len(sorted_counts)}
    cumulative = 0
    targets_remaining = sorted(coverage_targets)
    for i, c in enumerate(sorted_counts, start=1):
        cumulative += c
        frac = cumulative / total
        while targets_remaining and frac >= targets_remaining[0]:
            result[f"templates_for_{int(targets_remaining[0]*100)}pct"] = i
            targets_remaining.pop(0)
    for remaining in targets_remaining:
        result[f"templates_for_{int(remaining*100)}pct"] = len(sorted_counts)
    return result


def static_topk_candidate_recall(mask: torch.Tensor, block_size: int, fractions: list[float]) -> dict:
    """If a cheap, TOKEN-INDEPENDENT filter always nominates the top-K%
    most-frequently-active blocks (by marginal frequency across this
    sample) as candidates, what fraction of each token's TRUE active
    blocks does that recover? A real, if crude, lower bound on what a
    static (non-adaptive) candidate-block filter could achieve."""
    n = mask.shape[1]
    n_blocks = n // block_size
    blocks = mask.view(mask.shape[0], n_blocks, block_size).any(dim=-1)  # (tokens, n_blocks)
    block_freq = blocks.float().mean(dim=0)  # (n_blocks,)
    order = torch.argsort(block_freq, descending=True)
    out = {}
    for frac in fractions:
        k = max(1, int(round(n_blocks * frac)))
        candidate_idx = order[:k]
        candidate_mask = torch.zeros(n_blocks, dtype=torch.bool)
        candidate_mask[candidate_idx] = True
        true_active = blocks
        recovered = (true_active & candidate_mask.unsqueeze(0)).sum(dim=-1).float()
        true_count = true_active.sum(dim=-1).float().clamp(min=1)
        recall = (recovered / true_count).mean()
        out[f"top_{int(frac*100)}pct_blocks_recall"] = float(recall)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--block-sizes", type=str, default="16,32,64,128,256")
    parser.add_argument("--jaccard-distances", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--template-block-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device)
    config = BDHConfig(**payload["config"])
    model = BDH(config).to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    nh = config.n_head
    D = config.n_embd
    N = D * config.mlp_internal_dim_multiplier // nh
    n_total = nh * N
    last_round = config.n_layer - 1

    block_sizes = [int(x) for x in args.block_sizes.split(",")]
    distances = [int(x) for x in args.jaccard_distances.split(",")]

    all_last_round_x_masks = []  # each (T, n_total) for jaccard (needs per-sequence structure)
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
                    # x_sparse: (B, nh, T, N) -- encoder weight is PER-HEAD
                    # (model._w(model.encoder) has shape (nh, D, N)), so the
                    # head dimension is separate, not already folded into
                    # the neuron dimension. Permute so each token's full
                    # neuron vector (all heads concatenated) is contiguous.
                    mask = (x_sparse != 0).permute(0, 2, 1, 3).reshape(B, T, nh * N)  # (B, T, n_total)
                    for b in range(B):
                        all_last_round_x_masks.append(mask[b])  # (T, n_total)

                yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
                yKV = model.ln(yKV)
                y_latent = yKV @ model._w(model.encoder_v)
                y_sparse = F.relu(y_latent)
                g = x_sparse * y_sparse
                xy_sparse = model.drop(g)
                yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
                y = model.ln(yMLP)
                x = model.ln(x + y)

    flat_mask = torch.cat(all_last_round_x_masks, dim=0)  # (total_tokens, n_total)

    block_stats = [block_occupancy(flat_mask, bs) for bs in block_sizes]
    run_stats = run_length_stats(flat_mask)
    entropy = mask_entropy(flat_mask)
    templates = template_coverage(flat_mask, args.template_block_size, [0.90, 0.95, 0.99])
    recall = static_topk_candidate_recall(flat_mask, args.template_block_size, [0.10, 0.20, 0.30, 0.40, 0.50])

    jaccard_per_sequence = [nearby_token_jaccard(m, distances) for m in all_last_round_x_masks]
    jaccard_avg = {}
    for d in distances:
        vals = [j[str(d)] for j in jaccard_per_sequence if j[str(d)] is not None]
        jaccard_avg[str(d)] = sum(vals) / len(vals) if vals else None

    report = {
        "checkpoint": str(args.checkpoint),
        "scope_note": "computed on the LAST recurrent round only (steady-state activity), on the x-activation mask "
                       "(controls E_v/decoder exact skip -- see plan's x_density note). CertiGate 'certifiable-off "
                       "fraction' is NOT computed here -- that needs encoder-weight radius/centroid structure, a "
                       "separate later phase (plan section 10, Phase C-A).",
        "n_embd": D, "n_head": nh, "N_per_head": N, "n_total_neurons": n_total,
        "n_over_d_ratio": n_total / D,
        "last_round_index": last_round,
        "x_density_last_round": float(flat_mask.float().mean()),
        "block_occupancy": block_stats,
        "run_length_stats": run_stats,
        "mask_entropy_bits_per_neuron": entropy,
        "nearby_token_jaccard_by_distance": jaccard_avg,
        "template_coverage": templates,
        "static_topk_candidate_recall": recall,
        "total_tokens_sampled": flat_mask.shape[0],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
