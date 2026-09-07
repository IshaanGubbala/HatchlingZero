#!/usr/bin/env python3
"""HZ-Bench-100M Stage 0: solve for `d_model` that lands closest to a
target parameter count, holding every other HZ architectural setting
fixed (plans/Hatchling world.md section 0.8, user-directed: "preserve
the HZ architecture otherwise... have the script solve for d_model
that gets closest to 100M rather than guessing it"). Param count is
monotonically increasing in `d_model` (every major sub-module is a
Linear(d_model, ...) or scales with d_model) -- binary search over
real instantiated models, not a formula guess.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.language.tokenizer import NOVEL_LABELS  # noqa: E402


def count_params(d_model: int, memory_slots: int, workspace_slots: int, n_rounds_l1: int, vocab_size: int) -> int:
    model = HZLanguageModel(vocab_size=vocab_size, d_model=d_model, memory_slots=memory_slots,
                             workspace_slots=workspace_slots, n_rounds_l1=n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    return sum(p.numel() for p in model.parameters())


def solve_d_model(target_params: int, memory_slots: int, workspace_slots: int, n_rounds_l1: int,
                   vocab_size: int, lo: int = 64, hi: int = 4096) -> tuple[int, int]:
    # d_model must be even (value_dim = d_model // 2) and, per this repo's
    # existing HZ-Micro/HZ-Chat-Micro conventions, a multiple of 8 keeps
    # every attention head split clean -- round candidates to that grid.
    def round_grid(x: int) -> int:
        return max(8, round(x / 8) * 8)

    lo, hi = round_grid(lo), round_grid(hi)
    best_d, best_diff = lo, abs(count_params(lo, memory_slots, workspace_slots, n_rounds_l1, vocab_size) - target_params)
    while lo <= hi:
        mid = round_grid((lo + hi) // 2)
        n = count_params(mid, memory_slots, workspace_slots, n_rounds_l1, vocab_size)
        diff = abs(n - target_params)
        print(f"[size-solver] d_model={mid} -> n_params={n:,} (target {target_params:,}, diff {diff:,})", flush=True)
        if diff < best_diff:
            best_d, best_diff = mid, diff
        if n < target_params:
            lo = mid + 8
        elif n > target_params:
            hi = mid - 8
        else:
            return mid, n
    return best_d, count_params(best_d, memory_slots, workspace_slots, n_rounds_l1, vocab_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-params", type=int, default=100_000_000)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--workspace-slots", type=int, default=64)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    args = parser.parse_args()

    tok = ByteTokenizer()
    torch.manual_seed(0)
    d_model, n_params = solve_d_model(args.target_params, args.memory_slots, args.workspace_slots,
                                       args.n_rounds_l1, tok.vocab_size)
    pct_off = 100 * (n_params - args.target_params) / args.target_params
    print(f"\n[size-solver] BEST: d_model={d_model} n_params={n_params:,} "
          f"({pct_off:+.1f}% vs target {args.target_params:,})", flush=True)


if __name__ == "__main__":
    main()
