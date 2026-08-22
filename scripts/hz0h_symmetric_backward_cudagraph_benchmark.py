#!/usr/bin/env python3
"""Fresh-process CUDA benchmark: does wrapping d83db47's symmetric attention
backward in a CUDA graph (`torch.cuda.make_graphed_callables`) beat running
it eagerly?

Different axis from the broadcast attempt (which tried to remove a wasted
copy and found the GEMMs already dominated). This targets kernel-launch and
Python-dispatch overhead instead: the backward is bmm -> tril -> add ->
transpose -> bmm -> bmm -> sum, several small ops around GEMMs that are not
huge at T=256. CUDA graph replay issues all of them from a single pre-built
graph, skipping per-op CPU dispatch and (near-)eliminating launch latency.

This changes only execution scheduling, not the math, weights, or precision
-- allowed under this project's own exact-efficiency definition. It is also
mechanically distinct from the closed `torch.compile` lane: no TorchInductor
lowering is involved, only literal CUDA-stream capture/replay of the same
kernels eager execution would launch anyway.

Real constraint from `make_graphed_callables`: shapes/dtypes must be fixed
across replays (true here -- production training batches are fixed-shape),
and the graphed callable's inputs must be plain tensors, not something
requiring Python-side branching per call. It replays with fresh data each
call by copying into static input buffers, so varying *values* each step
(as a real training loop does) is exactly the intended use.
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
from reference.hz0h_bdh_torch import Attention, BDHConfig

ARMS = ("raw", "symmetric", "symmetric_graphed")


class _SymmetricModule(torch.nn.Module):
    def __init__(self, freqs: torch.Tensor):
        super().__init__()
        self.freqs = freqs

    def forward(self, q: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return bdh_symmetric_backward_attention(q, value, self.freqs)


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


def run_child(args):
    q, value, gradient, attention = make_inputs(args)

    if args.arm == "raw":
        def operation():
            return attention(q, q, value)
    elif args.arm == "symmetric":
        def operation():
            return bdh_symmetric_backward_attention(q, value, attention.freqs)
    else:
        module = _SymmetricModule(attention.freqs)
        sample_q = q.detach().clone().requires_grad_(True)
        sample_value = value.detach().clone().requires_grad_(True)
        graphed = torch.cuda.make_graphed_callables(module, (sample_q, sample_value))

        def operation():
            return graphed(q, value)

    for _ in range(args.warmup):
        operation().backward(gradient)
        q.grad = None
        value.grad = None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    output = None
    seconds_per_step_trials = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.steps):
            output = operation()
            output.backward(gradient)
            q.grad = None
            value.grad = None
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        seconds_per_step_trials.append(elapsed / args.steps)

    mean = sum(seconds_per_step_trials) / len(seconds_per_step_trials)
    variance = sum((t - mean) ** 2 for t in seconds_per_step_trials) / len(seconds_per_step_trials)
    stdev = variance ** 0.5
    result = {
        "arm": args.arm,
        "seconds_per_step_trials": seconds_per_step_trials,
        "seconds_per_step_mean": mean,
        "seconds_per_step_stdev": stdev,
        "tokens_per_second_mean": q.shape[0] * q.shape[2] / mean,
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

    module = _SymmetricModule(attention.freqs)
    sample_q = q.detach().clone().requires_grad_(True)
    sample_value = value.detach().clone().requires_grad_(True)
    graphed = torch.cuda.make_graphed_callables(module, (sample_q, sample_value))
    candidate = graphed(q, value)
    candidate.backward(gradient)
    return {
        "output_max_abs_error": float((candidate - reference).abs().max()),
        "q_gradient_max_abs_error": float((q.grad - q_grad_ref).abs().max()),
        "value_gradient_max_abs_error": float((value.grad - value_grad_ref).abs().max()),
        "finite": bool(torch.isfinite(candidate).all() and torch.isfinite(q.grad).all() and torch.isfinite(value.grad).all()),
    }


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
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
    ]


def run_parent(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    parity = verify_parity(args)
    results = {}
    with tempfile.TemporaryDirectory(prefix="hz0h_symmetric_cudagraph_") as directory:
        for arm in ARMS:
            child_out = Path(directory) / f"{arm}.json"
            subprocess.run(child_command(args, arm, child_out), check=True)
            results[arm] = json.loads(child_out.read_text(encoding="utf-8"))
    raw = results["raw"]
    symmetric = results["symmetric"]
    graphed = results["symmetric_graphed"]
    report = {
        "experiment_id": "bdh_symmetric_cudagraph_backward_v1",
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
            "identical math to d83db47's symmetric arm; wraps it in "
            "torch.cuda.make_graphed_callables to replay a captured CUDA "
            "graph instead of eager per-op dispatch. Execution scheduling "
            "only, not the Inductor/Dynamo path torch.compile uses."
        ),
        "fresh_subprocess_per_arm": True,
        "parity_vs_raw_graphed": parity,
        "arms": results,
        "symmetric_over_raw_throughput": symmetric["tokens_per_second_mean"] / raw["tokens_per_second_mean"],
        "graphed_over_raw_throughput": graphed["tokens_per_second_mean"] / raw["tokens_per_second_mean"],
        "graphed_over_symmetric_throughput": graphed["tokens_per_second_mean"] / symmetric["tokens_per_second_mean"],
        "graphed_over_symmetric_allocated_memory": (
            graphed["peak_memory_allocated_bytes"] / symmetric["peak_memory_allocated_bytes"]
        ),
        "stop_condition": "close if parity fails vs raw, or graphed does not beat symmetric on throughput",
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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
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
