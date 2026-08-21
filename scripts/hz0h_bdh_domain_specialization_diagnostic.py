#!/usr/bin/env python3
"""Tests a specific hypothesis raised 2026-08-21: does BDH show
DOMAIN-CONDITIONED neuron specialization -- different active `g_r`
support for code vs. math vs. documentation vs. terminal/debugging vs.
JSON/config -- when actually trained on genuinely diverse domains?

Part 11's `cross_token_support_jaccard` (see
`docs/restart/hz0h_inherited_choices_audit_results.md`) found LOW
cross-token support overlap (0.065-0.153 across depth) on held-out
data, and concluded a naive shared-identity block-sparse kernel isn't
supported. But that measurement pooled RANDOM token pairs from a
single, roughly homogeneous byte-level corpus with no domain
diversity -- it could not distinguish "support really is
per-token-random" from "support IS domain-conditioned, this dataset
just never had more than one domain to condition on." This script
trains on a real domain MIX (via `hz0h_bdh_domain_bytes_prep.py`'s
output, 5 genuinely distinct domains: code, documentation,
json_and_configuration, mathematical_and_structured,
terminal_and_debugging -- see that script's docstring for how the
existing BPE-tokenized `data/packed/external/` splits were bridged to
this project's byte-level vocab) and measures, per round:

- WITHIN-domain cross-token Jaccard: overlap between two DIFFERENT
  tokens from the SAME domain (e.g. two different code snippets).
- ACROSS-domain cross-token Jaccard: overlap between two tokens from
  DIFFERENT domains (e.g. a code token vs. a math token).

If domain-conditioned specialization is real, WITHIN should be
meaningfully higher than ACROSS. If it's close to equal, the earlier
"mostly per-token-random, not shared-identity" conclusion holds even
with real domain diversity to condition on.

Real, disclosed limits: local-scale training (small width, modest token
budget) -- a first-pass signal check, same caveat as every other
local-scale-first result in this project (Part 5, Part 6, Part 11's own
local-vs-CUDA gap). The domain-mixed training set itself is real but
modest (~21K windows from ~1500 packed source rows per domain, see
`hz0h_bdh_domain_bytes_prep.py`'s own caps) -- not the full external
corpus.

Never modifies `reference/hz0h_bdh_torch.py`,
`reference/hz0h_bdh_variable_depth_torch.py`,
`reference/hz0h_bdh_g_r_operator_diagnostic_torch.py`, or
`tokenizer/hz0a_tokenizer.py`.
"""
from __future__ import annotations

import argparse
import itertools
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
from scripts.hz0h_bdh_domain_bytes_prep import DOMAINS
from scripts.hz0h_bdh_g_r_operator_diagnostic import cross_domain_support_jaccard, cross_token_support_jaccard
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import read_batch


def collect_domain_reservoir(model, domain_path: Path, args, device, n_layer: int,
                              max_batches: int, reservoir_per_batch: int, seed: int) -> list[torch.Tensor]:
    """Returns a list of length `n_layer`, each entry a bounded
    `[<=max_batches*reservoir_per_batch, N]` tensor of real `g_r =
    u*v` samples pooled from THIS domain's own validation file only."""
    epochs = [0]
    per_round: list[list[torch.Tensor]] = [[] for _ in range(n_layer)]
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    with domain_path.open() as handle, torch.no_grad(), autocast_context(args, device):
        for _ in range(max_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx = data[:, :-1].contiguous()
            _, _, u_states, v_states = bdh_forward_with_g_r(model, idx, n_layer)
            for r, (u, v) in enumerate(zip(u_states, v_states)):
                flat = (u * v).reshape(-1, u.shape[-1])
                n = flat.shape[0]
                take = min(reservoir_per_batch, n)
                keep = torch.randperm(n, generator=generator)[:take]
                per_round[r].append(flat[keep].detach().cpu())
    return [torch.cat(chunks, dim=0) if chunks else torch.empty(0) for chunks in per_round]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains-dir", type=Path, default=Path("data/packed/domains"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-tokens", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--compile-training", action="store_true")
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="max-autotune")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--eval-batches-per-domain", type=int, default=15)
    parser.add_argument("--reservoir-per-batch", type=int, default=256)
    args = parser.parse_args()
    args.data = args.domains_dir / "mixed_train.jsonl"

    device = pick_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0)
    N = args.n_embd * args.mult // args.n_head
    print(f"=== training raw BDH on DOMAIN-MIXED data (n_embd={args.n_embd} mult={args.mult} "
          f"n_layer={args.n_layer} n_head={args.n_head}, N={N} per head) ===", flush=True)
    model, train_seconds = train_bdh(config, args, device, use_softmax_scaled=False)
    params = sum(p.numel() for p in model.parameters())
    print(f"[trained] params={params/1e6:.2f}M in {train_seconds:.0f}s", flush=True)

    print(f"=== collecting per-domain reservoirs ({args.eval_batches_per_domain} batches/domain) ===", flush=True)
    domain_reservoirs: dict[str, list[torch.Tensor]] = {}
    for name in DOMAINS:
        domain_path = args.domains_dir / f"{name}_val.jsonl"
        started = time.perf_counter()
        domain_reservoirs[name] = collect_domain_reservoir(
            model, domain_path, args, device, args.n_layer, args.eval_batches_per_domain,
            args.reservoir_per_batch, seed=args.seed,
        )
        print(f"[{name}] round0_samples={domain_reservoirs[name][0].shape[0]} "
              f"in {time.perf_counter()-started:.0f}s", flush=True)

    report = {"config": {"n_embd": args.n_embd, "mult": args.mult, "n_layer": args.n_layer, "n_head": args.n_head,
                          "N_per_head": N, "target_tokens": args.target_tokens, "seed": args.seed,
                          "domains": DOMAINS},
              "parameter_count": params, "train_seconds": train_seconds, "rounds": {}}

    for r in range(args.n_layer):
        within = []
        for name in DOMAINS:
            g = domain_reservoirs[name][r]
            if g.shape[0] < 2:
                continue
            within.append(cross_token_support_jaccard(g, n_pairs=1000, seed=args.seed)["mean_jaccard"])
        across = []
        for name_a, name_b in itertools.combinations(DOMAINS, 2):
            g_a, g_b = domain_reservoirs[name_a][r], domain_reservoirs[name_b][r]
            if g_a.shape[0] < 1 or g_b.shape[0] < 1:
                continue
            across.append(cross_domain_support_jaccard(g_a, g_b, n_pairs=1000, seed=args.seed)["mean_jaccard"])
        within_mean = sum(within) / len(within) if within else float("nan")
        across_mean = sum(across) / len(across) if across else float("nan")
        report["rounds"][str(r)] = {
            "within_domain_jaccard_mean": within_mean,
            "within_domain_jaccard_per_domain": dict(zip(DOMAINS, within)),
            "across_domain_jaccard_mean": across_mean,
            "across_domain_jaccard_per_pair": {
                f"{a}|{b}": v for (a, b), v in zip(itertools.combinations(DOMAINS, 2), across)
            },
        }
        ratio = within_mean / across_mean if across_mean and across_mean > 0 else float("nan")
        print(f"[round {r}] within_domain={within_mean:.4f} across_domain={across_mean:.4f} "
              f"ratio={ratio:.2f}x", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
