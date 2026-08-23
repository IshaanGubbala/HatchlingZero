#!/usr/bin/env python3
"""Does torch.compile(mode='reduce-overhead') work on BDH's streaming
decode step? Unlike the Transformer's KV-cache, BDH's per-layer states
never change shape across calls (fixed (B, n_head, N, D) the whole time,
by construction of the O(1)-state design) -- a much better CUDA-graph
candidate than a growing cache. The one real risk: bdh_stream_chunk takes
`start_position` as a plain Python int, which torch.compile's dynamo
treats as a specializing constant by default -- if that forces a
recompile every single step (since the value changes every call), this
would show up as compiled being much SLOWER than eager, not faster.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_chunk, bdh_stream_prefill_chunked, init_bdh_states
from reference.hz0h_bdh_stream_decode_step_graph_safe_torch import bdh_stream_decode_step_graph_safe_inplace


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
    prompt = torch.randint(args.vocab_size, (1, args.context_length), device=device)

    with torch.no_grad():
        states = init_bdh_states(model, prompt.shape[0], device=device)
        states, logits = bdh_stream_prefill_chunked(model, prompt, chunk_length=args.prefill_chunk_length, states=states)
        token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        def eager_decode(n):
            nonlocal states, token
            position = args.context_length
            for _ in range(n):
                states, logits = bdh_stream_chunk(model, states, token, start_position=position)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                position += 1

        for _ in range(4):
            eager_decode(1)
        eager_tps = time_decode(eager_decode, args.decode_tokens, device)

        compiled_step = torch.compile(bdh_stream_decode_step_graph_safe_inplace, mode="reduce-overhead")

        states2 = init_bdh_states(model, prompt.shape[0], device=device)
        states2, logits2 = bdh_stream_prefill_chunked(model, prompt, chunk_length=args.prefill_chunk_length, states=states2)
        token2 = torch.argmax(logits2[:, -1, :], dim=-1, keepdim=True)

        def compiled_decode(n):
            nonlocal token2
            position = torch.tensor(float(args.context_length), device=device)
            for _ in range(n):
                torch.compiler.cudagraph_mark_step_begin()
                logits2 = compiled_step(model, states2, token2, position)
                token2 = torch.argmax(logits2[:, -1, :], dim=-1, keepdim=True).clone()
                position = position + 1

        for _ in range(20):
            compiled_decode(1)
        compiled_tps = time_decode(compiled_decode, args.decode_tokens, device)

    print(f"bdh eager streaming decode: {eager_tps:.1f} tok/s")
    print(f"bdh compiled(reduce-overhead) streaming decode: {compiled_tps:.1f} tok/s")
    print(f"speedup: {compiled_tps/eager_tps:.2f}x")


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
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
