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

Trains a real raw BDH model with `train_bdh` from
`hz0h_bdh_combined_best_comparison.py` (same plain-attention recipe as
that script's `raw_bdh` arm), then STREAMS held-out validation batches
through `bdh_forward_with_g_r`, accumulating running statistics as it
goes rather than pooling full tensors -- see `StreamingRoundStats`'s
docstring for why the original pooling design was a real bug. Reports,
per recurrent round and pooled across rounds:

- density: fraction of `g_r` entries that are exactly zero (a ReLU*ReLU
  product, so exact zeros are real, not a threshold artifact) and the
  fraction above a small positive threshold (guards against near-zero
  noise being counted as "active").
- `f_x`/`f_y`/`f_xy`: `u=x_sparse`'s and `v=y_sparse`'s active fractions
  captured SEPARATELY (not just their product's zero pattern, which
  can't distinguish which factor was zero) -- `f_x` is known right
  after `E`, before attention/`E_v` even run, so it directly sizes the
  real, EXACT (not approximate) compute-skipping opportunity: `E_v`
  only needs evaluating on `u`'s support, and the decoder only needs
  the `f_xy` intersection.
- support Jaccard overlap between round r and r+1 (does the active set
  drift smoothly or churn completely each round?).
- top-k energy concentration: what fraction of a token-round's L2 energy
  is explained by its top-k entries, for a few k values as a fraction of
  N.
- effective rank (participation ratio `(sum(s))^2 / sum(s^2)` over the
  SVD singular values `s` of a bounded random reservoir of pooled `g_r`
  vectors -- the ONE statistic that genuinely needs actual sample
  vectors rather than running sums, so it alone gets a small fixed
  memory budget) both per-round and pooled across all rounds.
- `cross_token_support_jaccard` (added 2026-08-21): the crux question
  before building any block-sparse kernel. Low effective rank means the
  pooled `g_r` vectors jointly span a low-dimensional subspace -- it
  does NOT by itself mean different tokens share the same ACTIVE NEURON
  IDENTITIES (could be correlated activation on genuinely different
  supports). HIGH cross-token Jaccard (measured on random pairs of
  DIFFERENT tokens within the same round) means a STATIC, shared
  column-subset per round is exploitable with a real dense-GEMM, no
  gather needed; LOW cross-token Jaccard means the sparsity is real but
  token-specific, and naive per-token gather does NOT save real FLOPs
  (gathering a different weight slice per token costs as much as the
  matmul it replaces -- the same reason the earlier MoE-style router in
  this project's history lost on real wall-clock despite a real
  theoretical FLOP reduction).

Real, disclosed limits: local-scale results (n_embd=256) are a
first-pass signal check, matching this project's established pattern of
a cheap local prototype before a real CUDA-scale confirmation is worth
dispatching. A trained model's `g_r` statistics may not hold at
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
from scripts.hz0h_bdh_combined_best_comparison import autocast_context, train_bdh
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import read_batch


class StreamingRoundStats:
    """Real fix for a real crash (Windows dispatch, 2026-08-20): the
    original design pooled every batch's `u`/`v` tensors (even capped
    at `--max-samples-per-batch`) into growing Python lists, then
    `torch.cat`-ed everything at the end. At local scale (N=512) that
    topped out around 2.7GB, fine -- but at production width (N=4992)
    the capped-but-still-accumulating design reached ~12.5GB of pooled
    CPU tensors BEFORE the final `torch.cat`, which itself needs a
    second same-size allocation to run (old fragmented list + new
    contiguous tensor coexisting briefly) -- a real ~20-25GB peak
    against a machine with 16.7GB TOTAL system RAM. The process was
    silently killed (no traceback -- consistent with an OS-level
    OOM-kill, not a caught Python/CUDA exception).

    This class computes every statistic that reduces to a per-sample
    mean (density, f_x/f_y/f_xy, top-k energy, round-to-round Jaccard)
    as a running weighted average, one batch at a time -- no pooled
    tensor of samples is EVER materialized for these. Only
    `effective_rank`'s SVD genuinely needs real sample vectors (not
    just their mean), so ONLY that one gets a small fixed-size random
    reservoir (`svd_reservoir`, bounded by `svd_budget_per_batch *
    n_batches` regardless of N/batch_size/sequence_length)."""

    def __init__(self, svd_budget_per_batch: int, seed: int):
        self.count = 0
        self.exact_zero_count = 0.0
        self.above_threshold_count = 0.0
        self.active_u_count = 0.0
        self.active_v_count = 0.0
        self.active_both_count = 0.0
        self.topk_energy_weighted_sum: dict[int, float] = {}
        self.jaccard_weighted_sum = 0.0
        self.jaccard_count = 0
        self.svd_reservoir: list[torch.Tensor] = []
        self.svd_budget_per_batch = svd_budget_per_batch
        self.generator = torch.Generator().manual_seed(seed)

    def update(self, u: torch.Tensor, v: torch.Tensor, top_ks: list[int], threshold: float = 1e-6,
               next_round_u: torch.Tensor | None = None, next_round_v: torch.Tensor | None = None) -> None:
        """`u`/`v` shape `(B, n_head, T, N)` for THIS batch/round only --
        never retained after this call returns (caller discards them by
        going out of scope, GPU memory freed on the next batch)."""
        flat_u = u.reshape(-1, u.shape[-1])
        flat_v = v.reshape(-1, v.shape[-1])
        n = flat_u.shape[0]
        g = flat_u * flat_v

        # Real bug fixed here: each of these must be a PER-SAMPLE mean
        # over the N feature dim first (giving one fraction per sample,
        # shape (n,)), THEN summed over samples -- summing raw entry
        # counts over both dims and dividing by `self.count` (samples
        # only, not samples*N) silently produced counts in the hundreds
        # instead of fractions in [0, 1].
        active_u = flat_u > threshold
        active_v = flat_v > threshold
        self.exact_zero_count += float((g == 0).float().mean(dim=-1).sum())
        self.above_threshold_count += float((g > threshold).float().mean(dim=-1).sum())
        self.active_u_count += float(active_u.float().mean(dim=-1).sum())
        self.active_v_count += float(active_v.float().mean(dim=-1).sum())
        self.active_both_count += float((active_u & active_v).float().mean(dim=-1).sum())

        energy = g.pow(2)
        total_energy = energy.sum(dim=-1).clamp(min=1e-12)
        for k in top_ks:
            top_k = torch.topk(energy, k=min(k, g.shape[-1]), dim=-1).values.sum(dim=-1)
            batch_mean = float((top_k / total_energy).mean())
            self.topk_energy_weighted_sum[k] = self.topk_energy_weighted_sum.get(k, 0.0) + batch_mean * n

        if next_round_u is not None:
            flat_next_u = next_round_u.reshape(-1, next_round_u.shape[-1])
            flat_next_v = next_round_v.reshape(-1, next_round_v.shape[-1])
            next_g = flat_next_u * flat_next_v
            support_a = g > threshold
            support_b = next_g > threshold
            intersection = (support_a & support_b).sum(dim=-1).float()
            union = (support_a | support_b).sum(dim=-1).float().clamp(min=1.0)
            self.jaccard_weighted_sum += float((intersection / union).mean()) * n
            self.jaccard_count += n

        if self.svd_budget_per_batch > 0:
            take = min(self.svd_budget_per_batch, n)
            keep = torch.randperm(n, generator=self.generator)[:take]
            self.svd_reservoir.append(g[keep].detach().cpu())

        self.count += n

    def finalize(self, seed: int) -> dict:
        stats = {
            "exact_zero_fraction": self.exact_zero_count / self.count,
            "above_threshold_fraction": self.above_threshold_count / self.count,
            "f_x_active_fraction": self.active_u_count / self.count,
            "f_y_active_fraction": self.active_v_count / self.count,
            "f_xy_intersection_fraction": self.active_both_count / self.count,
            "top_k_energy_fraction": {
                f"k={k}": v_sum / self.count for k, v_sum in self.topk_energy_weighted_sum.items()
            },
            "samples_seen": self.count,
        }
        if self.jaccard_count > 0:
            stats["support_jaccard_to_next_round"] = self.jaccard_weighted_sum / self.jaccard_count
        if self.svd_reservoir:
            pooled = torch.cat(self.svd_reservoir, dim=0)
            stats["effective_rank"] = effective_rank(pooled, seed=seed)
            stats["cross_token_support_jaccard"] = cross_token_support_jaccard(pooled, seed=seed)
        return stats


def cross_token_support_jaccard(g: torch.Tensor, threshold: float = 1e-6, n_pairs: int = 2000, seed: int = 0) -> dict:
    """Real crux measurement (2026-08-21): a low `effective_rank` (SVD
    participation ratio) means the pooled `g_r` vectors jointly span a
    low-dimensional subspace -- it does NOT by itself mean different
    tokens share the same ACTIVE NEURON IDENTITIES. Those are genuinely
    different claims: low effective rank with LOW cross-token Jaccard
    would mean each token uses its own different small active set that
    happens to be jointly low-rank (correlated activation, not shared
    identity) -- exploiting that requires per-token-varying gather,
    which does NOT save real FLOPs naively (gathering a different
    weight slice per token costs as much as the matmul it replaces,
    the same reason the earlier MoE-style router lost). HIGH
    cross-token Jaccard would mean different tokens actually reuse the
    SAME small set of active neurons -- exploitable via one STATIC,
    shared column-subset per round, real GPU-friendly dense-GEMM
    savings, no gather needed. Measured here on random PAIRS of
    DIFFERENT samples within the SAME round's already-collected SVD
    reservoir (no new data collection)."""
    n = g.shape[0]
    if n < 2:
        return {"mean_jaccard": float("nan"), "pairs_used": 0}
    generator = torch.Generator().manual_seed(seed + 1)
    idx_a = torch.randint(0, n, (n_pairs,), generator=generator)
    idx_b = torch.randint(0, n, (n_pairs,), generator=generator)
    distinct = idx_a != idx_b
    idx_a, idx_b = idx_a[distinct], idx_b[distinct]
    support_a = g[idx_a] > threshold
    support_b = g[idx_b] > threshold
    intersection = (support_a & support_b).sum(dim=-1).float()
    union = (support_a | support_b).sum(dim=-1).float().clamp(min=1.0)
    return {"mean_jaccard": float((intersection / union).mean()), "pairs_used": int(idx_a.shape[0])}


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


def collect_streaming_stats(model, args, device, max_batches: int, top_ks: list[int],
                             svd_budget_per_batch: int) -> list[StreamingRoundStats]:
    """One `StreamingRoundStats` accumulator per recurrent round. Each
    batch's forward pass is real bf16 (see `autocast_context` below --
    a real bug fixed here too: this function used to call
    `bdh_forward_with_g_r` with NO autocast wrap, so it ran in fp32
    despite the model being trained under bf16 autocast, inflating GPU
    memory ~2-4x on top of the CPU-RAM bug this rewrite fixes)."""
    epochs = [0]
    accumulators = [StreamingRoundStats(svd_budget_per_batch, args.seed) for _ in range(model.config.n_layer)]
    model.eval()
    with args.validation_data.open() as handle, torch.no_grad(), autocast_context(args, device):
        for _ in range(max_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx = data[:, :-1].contiguous()
            _, _, u_states, v_states = bdh_forward_with_g_r(model, idx, model.config.n_layer)
            for r in range(len(u_states)):
                next_u = u_states[r + 1] if r + 1 < len(u_states) else None
                next_v = v_states[r + 1] if r + 1 < len(v_states) else None
                accumulators[r].update(u_states[r], v_states[r], top_ks, next_round_u=next_u, next_round_v=next_v)
    return accumulators


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
                         help="Held-out batches to stream through for g_r statistics after training.")
    parser.add_argument("--svd-reservoir-size", type=int, default=4000,
                         help="Total bounded sample budget (per round) for the effective-rank SVD "
                              "ONLY -- every other statistic is a running mean, no pooled tensor. "
                              "Real fix for the 2026-08-20 CUDA crash: the old design's cap still "
                              "pooled full tensors across all batches before computing anything.")
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

    top_ks = [max(1, int(N * frac)) for frac in args.top_k_fractions]
    svd_budget_per_batch = max(1, args.svd_reservoir_size // max(1, args.eval_batches))
    print(f"=== streaming {args.eval_batches} held-out batches through g_r diagnostic "
          f"(svd_reservoir<={svd_budget_per_batch * args.eval_batches} samples/round) ===", flush=True)
    started = time.perf_counter()
    accumulators = collect_streaming_stats(model, args, device, args.eval_batches, top_ks, svd_budget_per_batch)
    print(f"[collected] {accumulators[0].count} samples/round in {time.perf_counter()-started:.0f}s", flush=True)

    report = {"config": {"n_embd": args.n_embd, "mult": args.mult, "n_layer": args.n_layer,
                          "n_head": args.n_head, "N_per_head": N, "target_tokens": args.target_tokens,
                          "eval_batches": args.eval_batches, "svd_reservoir_size": args.svd_reservoir_size,
                          "seed": args.seed},
              "parameter_count": params, "train_seconds": train_seconds, "rounds": {}}

    top_k_label = f"k={top_ks[-1]}"
    for r, acc in enumerate(accumulators):
        stats = acc.finalize(seed=args.seed)
        report["rounds"][str(r)] = stats
        eff_rank = stats.get("effective_rank", {})
        cross_jaccard = stats.get("cross_token_support_jaccard", {})
        print(f"[round {r}] f_x={stats['f_x_active_fraction']:.3f} f_y={stats['f_y_active_fraction']:.3f} "
              f"f_xy={stats['f_xy_intersection_fraction']:.3f} "
              f"eff_rank={eff_rank.get('participation_ratio', float('nan')):.1f}"
              f"/{N} ({eff_rank.get('fraction_of_nominal_width', float('nan')):.1%}) "
              f"top-{top_ks[-1]}_energy={stats['top_k_energy_fraction'].get(top_k_label, float('nan')):.3f} "
              f"jaccard_to_next={stats.get('support_jaccard_to_next_round', float('nan')):.3f} "
              f"cross_token_jaccard={cross_jaccard.get('mean_jaccard', float('nan')):.3f}",
              flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
