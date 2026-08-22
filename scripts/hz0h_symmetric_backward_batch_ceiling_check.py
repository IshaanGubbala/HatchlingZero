#!/usr/bin/env python3
"""Find the real max batch size for the symmetric attention backward,
plain vs CUDA-graphed, at the production T=256/N=4992 shape.

The cudagraph benchmark (hz0h_symmetric_backward_cudagraph_benchmark.py)
showed peak *allocated* memory drops 34-42% under graph capture but peak
*reserved* -- the actual ceiling before OOM -- goes UP 15-39%. This directly
tests which one governs the real batch ceiling instead of inferring it from
a single-batch memory reading: doubles batch size in a fresh subprocess per
attempt (avoids allocator-state carryover) until OOM, for both arms.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def try_batch(args, arm: str, batch_size: int) -> dict:
    child = [
        sys.executable, str(Path(__file__).resolve()),
        "--child", "--arm", arm, "--batch-size", str(batch_size),
        "--sequence-length", str(args.sequence_length),
        "--n-embd", str(args.n_embd),
        "--heads", str(args.heads),
        "--multiplier", str(args.multiplier),
    ]
    proc = subprocess.run(child, capture_output=True, text=True)
    if proc.returncode != 0:
        oom = "OutOfMemoryError" in proc.stderr or "out of memory" in proc.stderr.lower()
        return {"batch_size": batch_size, "ok": False, "oom": oom, "stderr_tail": proc.stderr[-500:] if not oom else None}
    return {"batch_size": batch_size, "ok": True, **json.loads(proc.stdout.strip().splitlines()[-1])}


def find_ceiling(args, arm: str) -> dict:
    attempts = []
    batch_size = 1
    last_ok = None
    while batch_size <= args.max_batch_size:
        result = try_batch(args, arm, batch_size)
        attempts.append(result)
        print(f"[{arm}] batch={batch_size}: {'OK reserved=' + str(result.get('peak_memory_reserved_bytes')) if result['ok'] else ('OOM' if result.get('oom') else 'FAILED (non-OOM)')}", flush=True)
        if not result["ok"]:
            break
        last_ok = result
        batch_size *= 2
    return {"arm": arm, "max_working_batch_size": last_ok["batch_size"] if last_ok else 0, "attempts": attempts}


def run_child_real(args):
    from reference.hz0h_bdh_symmetric_backward_torch import bdh_symmetric_backward_attention
    from reference.hz0h_bdh_torch import Attention, BDHConfig

    device = torch.device("cuda")
    latent = args.n_embd * args.multiplier // args.heads
    q = torch.randn(args.batch_size, args.heads, args.sequence_length, latent, device=device, dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(args.batch_size, 1, args.sequence_length, args.n_embd, device=device, dtype=torch.bfloat16, requires_grad=True)
    gradient = torch.randn(args.batch_size, args.heads, args.sequence_length, args.n_embd, device=device, dtype=torch.bfloat16)
    attention = Attention(BDHConfig(n_embd=args.n_embd, n_head=args.heads, mlp_internal_dim_multiplier=args.multiplier)).to(device)

    if args.arm == "symmetric":
        def operation(qi, vi):
            return bdh_symmetric_backward_attention(qi, vi, attention.freqs)
    else:
        class _Module(torch.nn.Module):
            def __init__(self, freqs):
                super().__init__()
                self.freqs = freqs

            def forward(self, qi, vi):
                return bdh_symmetric_backward_attention(qi, vi, self.freqs)

        module = _Module(attention.freqs)
        sample_q = q.detach().clone().requires_grad_(True)
        sample_value = value.detach().clone().requires_grad_(True)
        graphed = torch.cuda.make_graphed_callables(module, (sample_q, sample_value))

        def operation(qi, vi):
            return graphed(qi, vi)

    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        operation(q, value).backward(gradient)
        q.grad = None
        value.grad = None
    torch.cuda.synchronize()
    print(json.dumps({
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--multiplier", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=1024)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=["symmetric", "symmetric_graphed"], default="symmetric", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.child:
        run_child_real(parsed)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        report = {arm: find_ceiling(parsed, arm) for arm in ("symmetric", "symmetric_graphed")}
        if parsed.out:
            parsed.out.parent.mkdir(parents=True, exist_ok=True)
            parsed.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
