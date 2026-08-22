#!/usr/bin/env python3
"""Fresh-process CUDA benchmark: raw vs symmetric (d83db47) vs
symmetric-broadcast (this session's follow-up) attention backward.

Same methodology as `hz0h_symmetric_backward_cuda_benchmark.py` (fresh
subprocess per arm, same seed/shape/warmup/steps), extended to a third arm
that removes the `value.expand().reshape()` materialized copy from the
symmetric backward (see `reference/hz0h_bdh_symmetric_backward_broadcast_torch.py`
for why that copy exists and why `torch.matmul`'s native broadcasting avoids
it). The comparison this script exists to make is specifically
symmetric-vs-symmetric-broadcast throughput and memory; raw is included only
as the already-established baseline.
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

from reference.hz0h_bdh_symmetric_backward_torch import bdh_symmetric_backward_attention
from reference.hz0h_bdh_symmetric_backward_broadcast_torch import (
    bdh_symmetric_broadcast_backward_attention,
)
from reference.hz0h_bdh_torch import Attention, BDHConfig

ARMS = ("raw", "symmetric", "broadcast")


def make_inputs(args):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent = args.n_embd * args.multiplier // args.heads
    q = torch.randn(
        args.batch_size, args.heads, args.sequence_length, latent,
        device=device, dtype=torch.bfloat16, generator=generator, requires_grad=True,
    )
    value = torch.randn(
        args.batch_size, 1, args.sequence_length, args.n_embd,
        device=device, dtype=torch.bfloat16, generator=generator, requires_grad=True,
    )
    gradient = torch.randn(
        args.batch_size, args.heads, args.sequence_length, args.n_embd,
        device=device, dtype=torch.bfloat16, generator=generator,
    )
    attention = Attention(
        BDHConfig(n_embd=args.n_embd, n_head=args.heads, mlp_internal_dim_multiplier=args.multiplier)
    ).to(device)
    return q, value, gradient, attention


def operation(arm, q, value, attention):
    if arm == "raw":
        return attention(q, q, value)
    if arm == "symmetric":
        return bdh_symmetric_backward_attention(q, value, attention.freqs)
    return bdh_symmetric_broadcast_backward_attention(q, value, attention.freqs)


def run_child(args):
    q, value, gradient, attention = make_inputs(args)
    for _ in range(args.warmup):
        operation(args.arm, q, value, attention).backward(gradient)
        q.grad = None
        value.grad = None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = None
    for _ in range(args.steps):
        output = operation(args.arm, q, value, attention)
        output.backward(gradient)
        q.grad = None
        value.grad = None
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "arm": args.arm,
        "seconds_per_step": elapsed / args.steps,
        "tokens_per_second": q.shape[0] * q.shape[2] * args.steps / elapsed,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "finite": bool(torch.isfinite(output).all()),
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


def verify_parity(args):
    q, value, gradient, attention = make_inputs(args)
    reference = attention(q, q, value)
    reference.backward(gradient)
    q_grad_ref = q.grad.detach().clone()
    value_grad_ref = value.grad.detach().clone()
    q.grad = None
    value.grad = None

    out = {}
    for arm, fn in (("symmetric", bdh_symmetric_backward_attention), ("broadcast", bdh_symmetric_broadcast_backward_attention)):
        candidate = fn(q, value, attention.freqs)
        candidate.backward(gradient)
        out[arm] = {
            "output_max_abs_error": float((candidate - reference).abs().max()),
            "q_gradient_max_abs_error": float((q.grad - q_grad_ref).abs().max()),
            "value_gradient_max_abs_error": float((value.grad - value_grad_ref).abs().max()),
            "finite": bool(torch.isfinite(candidate).all() and torch.isfinite(q.grad).all() and torch.isfinite(value.grad).all()),
        }
        q.grad = None
        value.grad = None
    return out


def child_command(args, arm, out):
    return [
        sys.executable, str(Path(__file__).resolve()),
        "--child", "--arm", arm, "--out", str(out),
        "--batch-size", str(args.batch_size),
        "--sequence-length", str(args.sequence_length),
        "--n-embd", str(args.n_embd),
        "--heads", str(args.heads),
        "--multiplier", str(args.multiplier),
        "--warmup", str(args.warmup),
        "--steps", str(args.steps),
        "--seed", str(args.seed),
    ]


def run_parent(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    parity = verify_parity(args)
    results = {}
    with tempfile.TemporaryDirectory(prefix="hz0h_symmetric_broadcast_") as directory:
        for arm in ARMS:
            child_out = Path(directory) / f"{arm}.json"
            subprocess.run(child_command(args, arm, child_out), check=True)
            results[arm] = json.loads(child_out.read_text(encoding="utf-8"))
    raw = results["raw"]
    symmetric = results["symmetric"]
    broadcast = results["broadcast"]
    report = {
        "experiment_id": "bdh_symmetric_broadcast_backward_v1",
        "scope": "attention forward+backward operator gate; not a full training claim",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "shape": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "n_embd": args.n_embd,
            "heads": args.heads,
            "latent_per_head": args.n_embd * args.multiplier // args.heads,
        },
        "algorithmic_difference": (
            "same (dS+dS.T)@Q identity as d83db47's symmetric arm; replaces "
            "value.expand().reshape() (a materialized (B,H,T,D) copy of value) "
            "with torch.matmul's native broadcasting over the head dimension"
        ),
        "fresh_subprocess_per_arm": True,
        "parity_vs_raw": parity,
        "arms": results,
        "symmetric_over_raw_throughput": symmetric["tokens_per_second"] / raw["tokens_per_second"],
        "broadcast_over_raw_throughput": broadcast["tokens_per_second"] / raw["tokens_per_second"],
        "broadcast_over_symmetric_throughput": broadcast["tokens_per_second"] / symmetric["tokens_per_second"],
        "broadcast_over_symmetric_allocated_memory": (
            broadcast["peak_memory_allocated_bytes"] / symmetric["peak_memory_allocated_bytes"]
        ),
        "broadcast_over_symmetric_reserved_memory": (
            broadcast["peak_memory_reserved_bytes"] / symmetric["peak_memory_reserved_bytes"]
        ),
        "stop_condition": "close if parity fails vs raw, or broadcast does not beat symmetric on both throughput and memory",
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
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--multiplier", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=list(ARMS), default="raw", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.child:
        run_child(parsed)
    else:
        run_parent(parsed)
