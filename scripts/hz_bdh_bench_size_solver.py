#!/usr/bin/env python3
"""HZ-Bench-100M, combined_best BDH variant: solve for `n_embd` that
lands closest to a target parameter count, holding the combined_best
recipe's own validated settings fixed (plans/Hatchling world.md section
0.8 update, user-directed: "use combined_best BDH" after Stage 0 found
HZLanguageModel's sequential per-token architecture severely memory/
throughput-bound even on a real 48GB L40S). Fixed per the audit
(`scripts/hz0h_bdh_combined_best_comparison.py`): `mlp_internal_dim_
multiplier=16` (not canonical 32), `n_layer=8`, `n_head=4` -- this
script only searches `n_embd`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig  # noqa: E402


def count_params(n_embd: int, n_layer: int, n_head: int, mult: int, vocab_size: int) -> int:
    config = BDHConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head,
                        mlp_internal_dim_multiplier=mult, vocab_size=vocab_size, dropout=0.0)
    model = BDH(config)
    return sum(p.numel() for p in model.parameters())


def solve_n_embd(target_params: int, n_layer: int, n_head: int, mult: int, vocab_size: int,
                  lo: int = 128, hi: int = 8192) -> tuple[int, int]:
    def round_grid(x: int) -> int:
        # n_embd must divide evenly by n_head (per-head split) -- round to that grid.
        return max(n_head, round(x / n_head) * n_head)

    lo, hi = round_grid(lo), round_grid(hi)
    best_d, best_diff = lo, abs(count_params(lo, n_layer, n_head, mult, vocab_size) - target_params)
    while lo <= hi:
        mid = round_grid((lo + hi) // 2)
        n = count_params(mid, n_layer, n_head, mult, vocab_size)
        diff = abs(n - target_params)
        print(f"[bdh-size-solver] n_embd={mid} -> n_params={n:,} (target {target_params:,}, diff {diff:,})",
              flush=True)
        if diff < best_diff:
            best_d, best_diff = mid, diff
        if n < target_params:
            lo = mid + n_head
        elif n > target_params:
            hi = mid - n_head
        else:
            return mid, n
    return best_d, count_params(best_d, n_layer, n_head, mult, vocab_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-params", type=int, default=100_000_000)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--mult", type=int, default=16, help="mlp_internal_dim_multiplier, combined_best recipe")
    parser.add_argument("--vocab-size", type=int, default=256, help="raw bytes, matching combined_best's own convention")
    args = parser.parse_args()

    n_embd, n_params = solve_n_embd(args.target_params, args.n_layer, args.n_head, args.mult, args.vocab_size)
    pct_off = 100 * (n_params - args.target_params) / args.target_params
    print(f"\n[bdh-size-solver] BEST: n_embd={n_embd} n_layer={args.n_layer} n_head={args.n_head} "
          f"mult={args.mult} n_params={n_params:,} ({pct_off:+.1f}% vs target {args.target_params:,})", flush=True)


if __name__ == "__main__":
    main()
