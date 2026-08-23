#!/usr/bin/env python3
"""Manual torch.cuda.graph() capture of BDH's in-place streaming decode
step -- bypasses torch.compile(reduce-overhead)'s conservative policy of
refusing to CUDA-graph functions with mutated inputs (measured: it falls
back to Inductor-fusion-only, 1.03x, not real graph replay).

Standard pattern for stateful CUDA-graph decode: fixed "static" input
buffers get copied into before each replay; the graph itself always
reads/writes the exact same memory addresses (the state tensors, mutated
in place via .add_(), and the static input/output buffers) -- this is
exactly what bdh_stream_decode_step_graph_safe_inplace already does, we
just capture it directly instead of routing through torch.compile's
safety net.
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
        # -- eager baseline --
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

        # -- manual CUDA graph capture --
        graph_states = init_bdh_states(model, prompt.shape[0], device=device)
        graph_states, logits3 = bdh_stream_prefill_chunked(model, prompt, chunk_length=args.prefill_chunk_length, states=graph_states)
        static_token = torch.argmax(logits3[:, -1, :], dim=-1, keepdim=True).clone()
        static_position = torch.tensor(float(args.context_length), device=device)

        # Warmup on a side stream, required before capture (PyTorch's own
        # documented requirement for torch.cuda.graph capture).
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                bdh_stream_decode_step_graph_safe_inplace(model, graph_states, static_token, static_position)
        torch.cuda.current_stream().wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits = bdh_stream_decode_step_graph_safe_inplace(model, graph_states, static_token, static_position)

        def graph_decode(n):
            nonlocal static_token
            pos = float(args.context_length)
            for _ in range(n):
                static_position.fill_(pos)
                graph.replay()
                static_token.copy_(torch.argmax(static_logits[:, -1, :], dim=-1, keepdim=True))
                pos += 1

        for _ in range(4):
            graph_decode(1)
        graph_tps = time_decode(graph_decode, args.decode_tokens, device)

    print(f"bdh eager streaming decode: {eager_tps:.1f} tok/s")
    print(f"bdh manual-cuda-graph streaming decode: {graph_tps:.1f} tok/s")
    print(f"speedup: {graph_tps/eager_tps:.2f}x")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--prefill-chunk-length", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
