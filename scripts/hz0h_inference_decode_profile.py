#!/usr/bin/env python3
"""Profile the Transformer's single-token KV-cache decode loop directly --
does it look launch-overhead-bound (many tiny kernels, GPU idle between
them) as the leading hypothesis for the suspiciously low 300-450 tok/s
single-sequence decode throughput measured on a real A40?
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM


def main(args):
    device = torch.device("cuda")
    torch.manual_seed(7)
    config = MatchedTransformerConfig({
        "vocab_size": args.vocab_size, "d_model": args.n_embd, "num_layers": args.transformer_layers,
        "num_heads": args.transformer_heads, "head_dim": args.n_embd // args.transformer_heads,
        "d_ff": args.d_ff, "use_rope": True,
    })
    model = MatchedTransformerLM(config).to(device=device, dtype=torch.bfloat16).eval()
    prompt = torch.randint(args.vocab_size, (1, args.context_length), device=device)

    with torch.no_grad():
        cache = model.new_kv_cache()
        logits = model(prompt, kv_cache=cache)
        token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        for _ in range(4):
            logits = model(token, kv_cache=cache)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize()

        cache = model.new_kv_cache()
        logits = model(prompt, kv_cache=cache)
        token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.decode_tokens):
                logits = model(token, kv_cache=cache)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            torch.cuda.synchronize()

    events = prof.key_averages()
    total_cuda_us = sum(max(getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0), 0) for e in events)
    total_cpu_us = sum(max(e.self_cpu_time_total, 0) for e in events)
    kernel_count = sum(e.count for e in events if (getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)) > 0)

    print(f"decode_tokens={args.decode_tokens}")
    print(f"total self CUDA time (us): {total_cuda_us:.1f}  -> avg per token: {total_cuda_us/args.decode_tokens:.1f}us")
    print(f"total self CPU time (us): {total_cpu_us:.1f}  -> avg per token: {total_cpu_us/args.decode_tokens:.1f}us")
    print(f"GPU-kernel-bearing op invocations: {kernel_count}  -> avg per token: {kernel_count/args.decode_tokens:.1f}")
    print()
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=9984)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
