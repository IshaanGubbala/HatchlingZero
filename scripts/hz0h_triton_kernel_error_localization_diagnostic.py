#!/usr/bin/env python3
"""Real error-localization diagnostic for the still-unresolved bf16 Triton
BDH kernel correctness failure (2026-08-16).

The input_precision="ieee" fix (reference/hz0h_bdh_triton_attention_torch.py,
commit 99af75c) was real -- it fixed a genuine TF32 rounding issue measured
by scripts/hz0h_triton_kernel_precision_diagnostic.py -- but a re-run of the
real bf16 correctness test came back with byte-identical failures to before
the fix. That's expected, not a contradiction: TF32 only affects tl.dot
calls with fp32-dtype inputs, and the real correctness test runs entirely
in bf16, so that fix never touched this code path at all.

This script narrows down WHERE the bf16 error is concentrated: per-row
max absolute/relative error (row = query sequence position). Real,
untested hypothesis: the oracle rounds to bf16 twice (after QR@KR.mT,
again after scores@V) while the kernel accumulates the whole reduction in
fp32 and rounds once at the end -- if true, the two computations are
BOTH "correct" in the sense of implementing the same math, but diverge by
bf16-rounding-chain noise that is proportionally largest at EARLY rows
(few accumulated terms, small magnitude, so a fixed absolute bf16-rounding
delta reads as a large relative error) -- the same pattern already
documented for the native tiled kernel in
docs/restart/hz0h_bdh_native_kernel_results.md. If errors are instead
concentrated at a specific LATE row, a tile boundary, or a masked
position that should be exactly zero, that would point to a genuine
logic bug instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import Attention, BDHConfig
from reference.hz0h_bdh_triton_attention_torch import bdh_triton_attention, triton_available

CASES = [
    (19, 32, 4, 32, 2, 17),
    (1, 16, 2, 16, 1, 8),
    (7, 64, 8, 16, 3, 33),
    (11, 32, 4, 8, 2, 65),
    (23, 128, 8, 32, 1, 256),
]


def run_case(seed, n_embd, n_head, mult, batch, seq_len):
    config = BDHConfig(n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=mult)
    attention = Attention(config).to(device="cuda", dtype=torch.bfloat16)
    attention.freqs = attention.freqs.to(torch.float32)
    N = n_embd * mult // n_head
    torch.manual_seed(seed)
    q = torch.randn(batch, n_head, seq_len, N, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, 1, seq_len, n_embd, device="cuda", dtype=torch.bfloat16)
    oracle = attention(q, q, v)
    kernel = bdh_triton_attention(q, v, attention.freqs)
    diff = (kernel - oracle).abs().float()

    flat_idx = int(diff.argmax().item())
    loc = list(torch.unravel_index(torch.tensor(flat_idx), diff.shape))
    b, h, t, d = (int(x) for x in loc)

    print(f"\ncase: n_embd={n_embd} n_head={n_head} mult={mult} batch={batch} seq_len={seq_len}")
    print(f"  overall max abs diff: {diff.max().item():.6f} at (b={b}, h={h}, t={t}, d={d})")
    print(f"    oracle value there: {oracle[b, h, t, d].item():.6f}")
    print(f"    kernel value there: {kernel[b, h, t, d].item():.6f}")
    print(f"    row t={t} oracle abs-max: {oracle[b, h, t, :].abs().max().item():.6f}")

    # Per-row (query position) max abs diff -- reveals whether error
    # concentrates at early (small-magnitude) rows or elsewhere.
    per_row_max = diff.amax(dim=(0, 1, 3))  # shape (T,)
    print("  per-row max abs diff (row: value), rows with diff > 0.01:")
    for t_idx in range(seq_len):
        val = per_row_max[t_idx].item()
        if val > 0.01:
            oracle_row_scale = oracle[:, :, t_idx, :].abs().max().item()
            print(f"    t={t_idx:3d}: diff={val:.6f}  oracle_row_abs_max={oracle_row_scale:.6f}  "
                  f"rel={val / max(oracle_row_scale, 1e-9):.4%}")

    # Are any exactly-zero (fully masked, e.g. row 0) oracle positions
    # non-zero in the kernel output? Real structural-bug check.
    zero_mask = oracle == 0
    if zero_mask.any():
        kernel_at_zero = kernel[zero_mask].abs().max().item()
        print(f"  max |kernel| at oracle-exactly-zero positions: {kernel_at_zero:.6f}")


def main() -> None:
    if not triton_available():
        print("SKIP: this machine has no CUDA/Triton -- run on the RTX3060 box.", file=sys.stderr)
        sys.exit(1)

    for seed, n_embd, n_head, mult, batch, seq_len in CASES:
        run_case(seed, n_embd, n_head, mult, batch, seq_len)


if __name__ == "__main__":
    main()
