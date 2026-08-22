#!/usr/bin/env python3
"""Float64 ground-truth parity check for BDH's symmetry-reduced attention
backward (`reference/hz0h_bdh_symmetric_backward_torch.py`).

Additive diagnostic -- does not modify or replace
`scripts/hz0h_symmetric_backward_cuda_benchmark.py`'s own gate. That script's
`verify_parity` only compares the "symmetric" arm against the "raw" arm, both
already in BF16: it answers "do they agree with each other," not "is either
one close to the true gradient." A BF16 rounding issue shared by both
formulations would pass that check silently. This script computes both
gradients in float64 (exact, to float64 precision) as ground truth and
reports each BF16 arm's OWN error against it, so a symmetric-specific
regression is distinguishable from ordinary BF16 rounding common to both.

Runs on CPU by default (float64 and BF16 matmul are both well-supported and
deterministic there, and no CUDA is needed to answer a numerics question --
only the separate throughput benchmark needs the real GPU). Pass --device
cuda to instead check BF16 behavior under real tensor-core accumulation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_symmetric_backward_torch import _BDHSymmetricBackward


def dense_attention(q: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Same formula as the oracle: tril(Q Q^T, -1) @ V, generic autograd."""
    scores = (q @ q.mT).tril(diagonal=-1)
    return scores @ value


def make_inputs(args, dtype, device):
    generator = torch.Generator().manual_seed(args.seed)
    latent = args.n_embd * args.multiplier // args.heads
    q64 = torch.randn(
        args.batch_size, args.heads, args.sequence_length, latent, generator=generator, dtype=torch.float64
    )
    value64 = torch.randn(
        args.batch_size, 1, args.sequence_length, args.n_embd, generator=generator, dtype=torch.float64
    )
    grad64 = torch.randn(
        args.batch_size, args.heads, args.sequence_length, args.n_embd, generator=generator, dtype=torch.float64
    )
    # .clone() forces a real copy even when dtype/device already match
    # float64/cpu, so the returned leaf never aliases q64/value64/grad64.
    q = q64.to(dtype=dtype, device=device).clone().requires_grad_(True)
    value = value64.to(dtype=dtype, device=device).clone().requires_grad_(True)
    grad = grad64.to(dtype=dtype, device=device).clone()
    return q, value, grad, q64, value64, grad64


def relative_error(candidate: torch.Tensor, truth: torch.Tensor) -> dict:
    diff = (candidate.double().cpu() - truth.cpu()).abs()
    truth_abs = truth.cpu().abs()
    max_abs = float(diff.max())
    l2_relative = float(diff.norm() / truth.norm().clamp_min(1e-30))
    denom = truth_abs.clamp_min(truth_abs.mean() * 1e-3 + 1e-12)
    max_relative = float((diff / denom).max())
    return {"max_abs_error": max_abs, "l2_relative_error": l2_relative, "max_relative_error_clamped": max_relative}


def run(args) -> dict:
    device = torch.device(args.device)

    # Ground truth: float64, always on CPU regardless of --device (fp64
    # matmul on CUDA has no tensor-core path and is not required here).
    q_truth, value_truth, grad_truth, _, _, _ = make_inputs(args, torch.float64, "cpu")
    out64 = dense_attention(q_truth, value_truth)
    out64.backward(grad_truth)
    q_grad_truth = q_truth.grad.detach().clone()
    value_grad_truth = value_truth.grad.detach().clone()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    q_bf, value_bf, grad_bf, *_ = make_inputs(args, dtype, device)

    # Raw arm: generic autograd through the same dense formula (two T^2*N
    # GEMMs for dQ, exactly what the closed lane's benchmark calls "raw").
    out_raw = dense_attention(q_bf, value_bf)
    out_raw.backward(grad_bf)
    q_grad_raw = q_bf.grad.detach().clone()
    value_grad_raw = value_bf.grad.detach().clone()

    # Symmetric arm: the actual custom autograd.Function under test.
    q_bf2 = q_bf.detach().clone().requires_grad_(True)
    value_bf2 = value_bf.detach().clone().requires_grad_(True)
    out_sym = _BDHSymmetricBackward.apply(q_bf2, value_bf2)
    out_sym.backward(grad_bf)
    q_grad_sym = q_bf2.grad.detach().clone()
    value_grad_sym = value_bf2.grad.detach().clone()

    report = {
        "experiment_id": "bdh_symmetric_dense_backward_v1_ground_truth_check",
        "scope": "does NOT replace the closed-lane raw-vs-symmetric BF16 comparison; "
        "checks each arm independently against a float64 reference",
        "device": args.device,
        "dtype": args.dtype,
        "shape": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "n_embd": args.n_embd,
            "heads": args.heads,
            "latent_per_head": args.n_embd * args.multiplier // args.heads,
        },
        "q_gradient": {
            "raw_vs_truth": relative_error(q_grad_raw, q_grad_truth),
            "symmetric_vs_truth": relative_error(q_grad_sym, q_grad_truth),
        },
        "value_gradient": {
            "raw_vs_truth": relative_error(value_grad_raw, value_grad_truth),
            "symmetric_vs_truth": relative_error(value_grad_sym, value_grad_truth),
        },
        "output": {
            "raw_vs_truth": relative_error(out_raw, out64),
            "symmetric_vs_truth": relative_error(out_sym, out64),
        },
    }
    q_ratio = (
        report["q_gradient"]["symmetric_vs_truth"]["l2_relative_error"]
        / max(report["q_gradient"]["raw_vs_truth"]["l2_relative_error"], 1e-30)
    )
    report["symmetric_over_raw_q_gradient_l2_relative_error_ratio"] = q_ratio
    report["verdict"] = (
        "symmetric error is within 2x of raw's own BF16 error against ground truth "
        "-- consistent with ordinary BF16 rounding shared by both formulations, not a "
        "symmetric-specific bug"
        if q_ratio < 2.0
        else "symmetric error is more than 2x raw's own BF16 error against ground truth "
        "-- warrants investigation before treating the parity gate as clean"
    )
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--multiplier", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = run(parsed)
    parsed.out.parent.mkdir(parents=True, exist_ok=True)
    parsed.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
