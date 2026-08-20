#!/usr/bin/env python3
"""Real measurement of whether BDH's REALIZED per-round operator gate

    g_r = x_sparse_r (elementwise*) y_sparse_r

(see `reference/hz0h_bdh_g_r_operator_diagnostic_torch.py`'s docstring
for the full derivation: `g_r` is the coefficient vector selecting which
of the `N`-per-head rank-1 operators `e_n d_n^T` are active for a given
token/round) is anywhere near as sparse/low-dimensional as its nominal
width `N = D * mult // n_head` -- i.e. whether "the architecture is
discovering sparsity after paying for it," the load-bearing premise
behind every proposed follow-up (recurrent active-set BDH, operator-bank
sketching) in the 2026-08-20 discussion that motivated this script.

Trains a real (small, local-scale) raw BDH model with `train_bdh` from
`hz0h_bdh_combined_best_comparison.py` (same plain-attention recipe as
that script's `raw_bdh` arm, just a smaller width/token-budget so it
trains in minutes on this Mac instead of hours), then runs
`bdh_forward_with_g_r` over held-out validation batches and reports,
per recurrent round and pooled across rounds:

- density: fraction of `g_r` entries that are exactly zero (a ReLU*ReLU
  product, so exact zeros are real, not a threshold artifact) and the
  fraction above a small positive threshold (guards against near-zero
  noise being counted as "active").
- support Jaccard overlap between round r and r+1 (does the active set
  drift smoothly or churn completely each round?).
- top-k energy concentration: what fraction of a token-round's L2 energy
  is explained by its top-k entries, for a few k values as a fraction of
  N.
- effective rank (participation ratio `(sum(s))^2 / sum(s^2)` over the
  SVD singular values `s` of a `[samples, N]` matrix of pooled `g_r`
  vectors) both per-round and pooled across all rounds.

Real, disclosed limits: SMALL local scale (not the real 300M/n_embd=2496
production shape used elsewhere in this project's CUDA work) -- this is
a first-pass signal check, matching this project's established pattern
of a cheap local prototype before a real CUDA-scale confirmation is
worth dispatching. A trained model's `g_r` statistics may not hold at
production width; treat this script's numbers as a hypothesis check,
not a final answer, same caveat this project attaches to every other
local-scale-first result (Part 5, Part 6's original prototype).

Never modifies `reference/hz0h_bdh_torch.py`,
`reference/hz0h_bdh_variable_depth_torch.py`, or
`reference/hz0h_bdh_g_r_operator_diagnostic_torch.py`.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_g_r_operator_diagnostic_torch import bdh_forward_with_g_r
from reference.hz0h_bdh_torch import BDHConfig
from scripts.hz0h_bdh_combined_best_comparison import train_bdh
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import read_batch


def collect_g_states(model, args, device, max_batches: int) -> list[torch.Tensor]:
    """Returns a list of length `n_layer`, each entry a `[samples, N]`
    tensor pooling `g_r` across ALL held-out batches/tokens/heads for
    that recurrent round -- `samples = max_batches * batch_size *
    sequence_length * n_head`."""
    epochs = [0]
    per_round: list[list[torch.Tensor]] = [[] for _ in range(model.config.n_layer)]
    model.eval()
    with args.validation_data.open() as handle, torch.no_grad():
        for _ in range(max_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx = data[:, :-1].contiguous()
            _, _, g_states = bdh_forward_with_g_r(model, idx, model.config.n_layer)
            for r, g in enumerate(g_states):
                # g: (B, n_head, T, N) -> (B*n_head*T, N), pool heads/tokens/batch together
                per_round[r].append(g.reshape(-1, g.shape[-1]).cpu())
    return [torch.cat(chunks, dim=0) for chunks in per_round]


def density_stats(g: torch.Tensor, threshold: float = 1e-6) -> dict:
    exact_zero_frac = float((g == 0).float().mean())
    above_threshold_frac = float((g > threshold).float().mean())
    return {"exact_zero_fraction": exact_zero_frac, "above_threshold_fraction": above_threshold_frac}


def support_jaccard(g_a: torch.Tensor, g_b: torch.Tensor, threshold: float = 1e-6) -> float:
    """Mean per-sample Jaccard overlap of {n: g[n] > threshold} between
    two same-shape pooled round matrices (paired row-for-row -- same
    token/head/batch position at round r vs round r+1)."""
    support_a = g_a > threshold
    support_b = g_b > threshold
    intersection = (support_a & support_b).sum(dim=-1).float()
    union = (support_a | support_b).sum(dim=-1).float()
    safe_union = union.clamp(min=1.0)
    return float((intersection / safe_union).mean())


def top_k_energy_fraction(g: torch.Tensor, k: int) -> float:
    """Mean per-sample fraction of L2 energy (sum of squares) explained
    by the top-`k` entries, over `g`'s pooled samples."""
    energy = g.pow(2)
    total = energy.sum(dim=-1).clamp(min=1e-12)
    top_k = torch.topk(energy, k=min(k, g.shape[-1]), dim=-1).values.sum(dim=-1)
    return float((top_k / total).mean())


def effective_rank(g: torch.Tensor, max_samples: int = 4000, seed: int = 0) -> dict:
    """Participation ratio `(sum(s))^2 / sum(s^2)` over SVD singular
    values of a randomly-subsampled `[samples, N]` slice -- a real
    (not eyeballed) scalar summary of how many dimensions the pooled
    `g_r` vectors actually span, independent of the nominal width."""
    n = g.shape[0]
    if n > max_samples:
        generator = torch.Generator().manual_seed(seed)
        keep = torch.randperm(n, generator=generator)[:max_samples]
        g = g[keep]
    centered = g - g.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered.double())
    s2 = singular_values.pow(2)
    participation_ratio = float(singular_values.sum().pow(2) / s2.sum().clamp(min=1e-12))
    nominal_width = g.shape[-1]
    return {
        "participation_ratio": participation_ratio,
        "nominal_width": nominal_width,
        "fraction_of_nominal_width": participation_ratio / nominal_width,
        "samples_used": int(g.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-tokens", type=int, default=300_000,
                         help="Small on purpose -- this is a local first-pass signal check, "
                              "not a production-scale training run. See module docstring.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--compile-training", action="store_true")
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="max-autotune")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=20,
                         help="Held-out batches to pool g_r statistics over after training.")
    parser.add_argument("--top-k-fractions", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.25],
                         help="Report top-k energy concentration at these fractions of N.")
    args = parser.parse_args()

    device = pick_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0)
    N = args.n_embd * args.mult // args.n_head
    print(f"=== training raw BDH (n_embd={args.n_embd} mult={args.mult} n_layer={args.n_layer} "
          f"n_head={args.n_head}, N={N} per head) for g_r diagnostic ===", flush=True)
    model, train_seconds = train_bdh(config, args, device, use_softmax_scaled=False)
    params = sum(p.numel() for p in model.parameters())
    print(f"[trained] params={params/1e6:.2f}M in {train_seconds:.0f}s", flush=True)

    print(f"=== collecting g_r over {args.eval_batches} held-out batches ===", flush=True)
    started = time.perf_counter()
    per_round = collect_g_states(model, args, device, args.eval_batches)
    print(f"[collected] {per_round[0].shape[0]} samples/round in {time.perf_counter()-started:.0f}s", flush=True)

    report = {"config": {"n_embd": args.n_embd, "mult": args.mult, "n_layer": args.n_layer,
                          "n_head": args.n_head, "N_per_head": N, "target_tokens": args.target_tokens,
                          "eval_batches": args.eval_batches, "seed": args.seed},
              "parameter_count": params, "train_seconds": train_seconds, "rounds": {}}

    top_ks = [max(1, int(N * frac)) for frac in args.top_k_fractions]
    for r, g in enumerate(per_round):
        stats = density_stats(g)
        stats["top_k_energy_fraction"] = {
            f"k={k}({frac:.0%}N)": top_k_energy_fraction(g, k) for k, frac in zip(top_ks, args.top_k_fractions)
        }
        stats["effective_rank"] = effective_rank(g, seed=args.seed)
        if r + 1 < len(per_round):
            stats["support_jaccard_to_next_round"] = support_jaccard(g, per_round[r + 1])
        report["rounds"][str(r)] = stats
        print(f"[round {r}] exact_zero={stats['exact_zero_fraction']:.3f} "
              f"eff_rank={stats['effective_rank']['participation_ratio']:.1f}"
              f"/{N} ({stats['effective_rank']['fraction_of_nominal_width']:.1%}) "
              f"top-{top_ks[-1]}_energy={stats['top_k_energy_fraction'][f'k={top_ks[-1]}({args.top_k_fractions[-1]:.0%}N)']:.3f}",
              flush=True)

    pooled = torch.cat(per_round, dim=0)
    report["pooled_across_all_rounds"] = {
        **density_stats(pooled),
        "effective_rank": effective_rank(pooled, seed=args.seed),
        "top_k_energy_fraction": {
            f"k={k}({frac:.0%}N)": top_k_energy_fraction(pooled, k) for k, frac in zip(top_ks, args.top_k_fractions)
        },
    }
    print(f"[pooled] eff_rank={report['pooled_across_all_rounds']['effective_rank']['participation_ratio']:.1f}"
          f"/{N} ({report['pooled_across_all_rounds']['effective_rank']['fraction_of_nominal_width']:.1%})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
