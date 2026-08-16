#!/usr/bin/env python3
"""Real diagnostic for the 5/5 Triton BDH attention correctness failures
(tests/reference/test_hz0h_bdh_triton_attention_torch.py, real CUDA run,
2026-08-16): isolate whether the divergence is a genuine kernel logic bug
or an expected precision-order artifact.

Real, disclosed hypothesis being tested: the oracle (reference/hz0h_bdh_torch.py
Attention.forward) rounds to bf16 TWICE -- once after `QR @ KR.mT`, again
after `scores @ V` -- because both are separate bf16 matmuls. The Triton
kernel (reference/hz0h_bdh_triton_attention_torch.py) accumulates the whole
QK^T-then-V reduction in fp32 inside one fused kernel and only rounds once,
at the final store. If that's the real cause, the SAME two implementations
run entirely in fp32 (no bf16 rounding anywhere) should agree tightly
(this repo's usual fp32 bar, atol/rtol=1e-3) even though the bf16 versions
don't -- that would mean the kernel's math is correct and the bf16 test's
tolerance was miscalibrated (same real pattern already found once this
session in the native tiled kernel, see docs/restart/hz0h_bdh_native_kernel_results.md).

If the fp32 versions ALSO diverge by anywhere near the bf16-observed
magnitude, that rules out precision-order and points to a real logic bug
in the Triton kernel itself.

This is diagnostic only -- it does not change the pass/fail bar of the
real correctness test.
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


def run_case(seed, n_embd, n_head, mult, batch, seq_len, dtype):
    config = BDHConfig(n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=mult)
    attention = Attention(config).to(device="cuda", dtype=dtype)
    attention.freqs = attention.freqs.to(torch.float32)
    N = n_embd * mult // n_head
    torch.manual_seed(seed)
    q = torch.randn(batch, n_head, seq_len, N, device="cuda", dtype=dtype)
    v = torch.randn(batch, 1, seq_len, n_embd, device="cuda", dtype=dtype)
    oracle = attention(q, q, v)
    kernel = bdh_triton_attention(q, v, attention.freqs)
    diff = (kernel - oracle).abs()
    max_diff = diff.max().item()
    oracle_scale = oracle.abs().max().item()
    return max_diff, oracle_scale


def main() -> None:
    if not triton_available():
        print("SKIP: this machine has no CUDA/Triton -- run on the RTX3060 box.", file=sys.stderr)
        sys.exit(1)

    print(f"{'case':<45} {'bf16 max diff':>15} {'bf16 scale':>12} {'fp32 max diff':>15} {'fp32 scale':>12}")
    for seed, n_embd, n_head, mult, batch, seq_len in CASES:
        bf16_diff, bf16_scale = run_case(seed, n_embd, n_head, mult, batch, seq_len, torch.bfloat16)
        fp32_diff, fp32_scale = run_case(seed, n_embd, n_head, mult, batch, seq_len, torch.float32)
        label = f"n_embd={n_embd} n_head={n_head} mult={mult} B={batch} T={seq_len}"
        print(f"{label:<45} {bf16_diff:>15.4f} {bf16_scale:>12.4f} {fp32_diff:>15.6f} {fp32_scale:>12.4f}")


if __name__ == "__main__":
    main()
