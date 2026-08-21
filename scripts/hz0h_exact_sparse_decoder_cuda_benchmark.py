#!/usr/bin/env python3
"""Benchmark exact sparse decoder training math at the real 300M shape.

Each arm runs in a fresh subprocess so the dense arm cannot influence sparse
library algorithm selection through co-resident allocations.  The benchmark
times forward and backward, verifies outputs and gradients against dense
matmul, and reports peak allocated/reserved CUDA memory as JSON.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_exact_sparse_decoder_torch import exact_sparse_decoder_mm


def synchronize() -> None:
    torch.cuda.synchronize()


def make_inputs(args, device):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    rows = args.batch_size * args.sequence_length
    latent_width = args.n_embd * args.multiplier
    values = torch.randn(rows, latent_width, device=device, dtype=torch.bfloat16, generator=generator)
    keep = torch.rand(rows, latent_width, device=device, generator=generator) < args.active_fraction
    values = (values.relu() * keep).detach().requires_grad_(True)
    weight = torch.randn(
        latent_width,
        args.n_embd,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02).requires_grad_(True)
    gradient = torch.randn(
        rows,
        args.n_embd,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    return values, weight, gradient


def operation(arm, values, weight):
    if arm == "dense":
        return values @ weight
    return exact_sparse_decoder_mm(values, weight, layout=arm)


def run_child(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    values, weight, gradient = make_inputs(args, device)

    for _ in range(args.warmup):
        output = operation(args.arm, values, weight)
        output.backward(gradient)
        values.grad = None
        weight.grad = None
    synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = None
    for _ in range(args.steps):
        output = operation(args.arm, values, weight)
        output.backward(gradient)
        values.grad = None
        weight.grad = None
    synchronize()
    seconds = time.perf_counter() - started

    nonzero = int(torch.count_nonzero(values).item())
    total = values.numel()
    result = {
        "arm": args.arm,
        "seconds": seconds,
        "seconds_per_step": seconds / args.steps,
        "tokens_per_second": args.batch_size * args.sequence_length * args.steps / seconds,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "finite": bool(torch.isfinite(output).all()),
        "nonzero_fraction": nonzero / total,
        "theoretical_decoder_multiply_fraction": nonzero / total,
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def child_command(args, arm, out):
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--arm", arm,
        "--out", str(out),
        "--batch-size", str(args.batch_size),
        "--sequence-length", str(args.sequence_length),
        "--n-embd", str(args.n_embd),
        "--multiplier", str(args.multiplier),
        "--active-fraction", str(args.active_fraction),
        "--warmup", str(args.warmup),
        "--steps", str(args.steps),
        "--seed", str(args.seed),
    ]


def verify_parity(args) -> dict:
    device = torch.device("cuda")
    values, weight, gradient = make_inputs(args, device)
    dense = values @ weight
    dense.backward(gradient)
    dense_values_grad = values.grad.detach().clone()
    dense_weight_grad = weight.grad.detach().clone()
    values.grad = None
    weight.grad = None

    reports = {}
    active = values != 0
    for layout in ("coo", "csr"):
        try:
            candidate = exact_sparse_decoder_mm(values, weight, layout=layout)
            candidate.backward(gradient)
            reports[layout] = {
                "supported": True,
                "output_max_abs_error": float((candidate - dense).abs().max()),
                "active_input_gradient_max_abs_error": float(
                    (values.grad[active] - dense_values_grad[active]).abs().max()
                ),
                "omitted_input_gradients_are_zero": bool(torch.count_nonzero(values.grad[~active]) == 0),
                "weight_gradient_max_abs_error": float((weight.grad - dense_weight_grad).abs().max()),
                "finite": bool(
                    torch.isfinite(candidate).all()
                    and torch.isfinite(values.grad).all()
                    and torch.isfinite(weight.grad).all()
                ),
            }
        except Exception as error:
            reports[layout] = {"supported": False, "error": repr(error)}
        values.grad = None
        weight.grad = None
    return reports


def run_parent(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    parity = verify_parity(args)
    arms = ["dense"] + [layout for layout in ("coo", "csr") if parity[layout]["supported"]]
    results = {}
    with tempfile.TemporaryDirectory(prefix="hz0h_sparse_decoder_") as directory:
        for arm in arms:
            child_out = Path(directory) / f"{arm}.json"
            subprocess.run(child_command(args, arm, child_out), check=True)
            results[arm] = json.loads(child_out.read_text(encoding="utf-8"))

    dense = results["dense"]
    for arm, result in results.items():
        result["throughput_over_dense"] = result["tokens_per_second"] / dense["tokens_per_second"]
        result["allocated_memory_over_dense"] = (
            result["peak_memory_allocated_bytes"] / dense["peak_memory_allocated_bytes"]
        )
    report = {
        "experiment_id": "exact_sparse_decoder_vendor_spmm_v1",
        "scope": "decoder operator forward+backward microbenchmark; not an end-to-end training claim",
        "novelty_vs_closed_sparse_lanes": (
            "skips only realized exact ReLU-product zeros via vendor CUDA COO/CSR SpMM; "
            "no router, top-k, pruning, threshold, gather_mm, or changed model math"
        ),
        "stop_condition": (
            "close vendor-SpMM lane if neither parity-passing COO nor CSR beats dense "
            "at the production shape; do not repeat with cosmetic tuning"
        ),
        "methodological_guards": {
            "fresh_subprocess_per_timed_arm": True,
            "same_seed_and_tensors": True,
            "cuda_synchronized_timing": True,
            "forward_and_backward_in_scope": True,
            "optimizer_step_in_scope": False,
            "full_model_training_in_scope": False,
        },
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "shape": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "rows": args.batch_size * args.sequence_length,
            "n_embd": args.n_embd,
            "decoder_input_width": args.n_embd * args.multiplier,
            "active_fraction_requested": args.active_fraction,
        },
        "parity": parity,
        "arms": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--multiplier", type=int, default=16)
    parser.add_argument("--active-fraction", type=float, default=0.12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=["dense", "coo", "csr"], default="dense", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.child:
        print(json.dumps(run_child(parsed), indent=2))
    else:
        run_parent(parsed)
