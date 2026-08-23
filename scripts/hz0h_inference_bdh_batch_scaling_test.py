#!/usr/bin/env python3
"""Does BDH streaming decode throughput scale with batch size? Manual
CUDA graph capture proved launch overhead isn't the real bottleneck at
batch=1 (1.01x speedup, effectively none) -- this tests the remaining
real hypothesis: batch=1 starves the GPU of parallelism regardless of
how kernels get launched, and batching multiple sequences should recover
real throughput, the same way training's batch=8 already does.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_chunk, bdh_stream_prefill_chunked, init_bdh_states


def time_decode(fn, n, device):
    torch.cuda.synchronize()
    started = time.perf_counter()
    fn(n)
    torch.cuda.synchronize()
    return n / (time.perf_counter() - started)


def main(args):
    device = torch.device("cuda")
    torch.manual_seed(7)
    config = BDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size, dropout=0.0,
    )
    model = BDH(config).to(device=device, dtype=torch.bfloat16).eval()

    for batch_size in [int(b) for b in args.batch_sizes.split(",")]:
        prompt = torch.randint(args.vocab_size, (batch_size, args.context_length), device=device)
        with torch.no_grad():
            states = init_bdh_states(model, batch_size, device=device)
            states, logits = bdh_stream_prefill_chunked(model, prompt, chunk_length=args.prefill_chunk_length, states=states)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            def decode(n):
                nonlocal states, token
                position = args.context_length
                for _ in range(n):
                    states, logits = bdh_stream_chunk(model, states, token, start_position=position)
                    token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    position += 1

            for _ in range(4):
                decode(1)
            torch.cuda.reset_peak_memory_stats()
            seq_per_second = time_decode(decode, args.decode_tokens, device)
            tokens_per_second = seq_per_second * batch_size
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"batch={batch_size}: {seq_per_second:.1f} steps/s -> {tokens_per_second:.1f} tok/s total, peak_mem={peak_gb:.2f}GB")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--prefill-chunk-length", type=int, default=1024)
    parser.add_argument("--batch-sizes", type=str, default="1,8,32,64")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
