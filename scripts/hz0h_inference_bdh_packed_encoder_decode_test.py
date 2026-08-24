#!/usr/bin/env python3
"""Does the packed-encoder layout (this session's training-side +4.19%
win) do anything for decode speed? Real test, not assumed: the manual-
CUDA-graph result already proved decode's bottleneck is the ~1.6GB
per-layer state I/O, not the encoder weight (~200MB) -- so the honest
prior is "probably small," but this measures it directly, eager and
under manual CUDA graph capture (the true zero-launch-overhead
baseline), against the original oracle broadcast-encoder decode step.

Prefill always runs through the oracle (bdh_stream_prefill_chunked uses
model.encoder's broadcast path internally, which PackedEncoderBDH
doesn't have -- it deletes .encoder). Same seed => same weights, so the
resulting states are valid starting points for every arm; each arm gets
its own clone so none of them share mutable state.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_packed_encoder_torch import PackedEncoderBDH
from reference.hz0h_bdh_stream_decode_step_graph_safe_torch import bdh_stream_decode_step_graph_safe_inplace
from reference.hz0h_bdh_stream_decode_step_packed_encoder_torch import bdh_stream_decode_step_packed_encoder_inplace
from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_chunk, bdh_stream_prefill_chunked, init_bdh_states


def time_decode(fn, n, device):
    torch.cuda.synchronize()
    started = time.perf_counter()
    fn(n)
    torch.cuda.synchronize()
    return n / (time.perf_counter() - started)


def eager_baseline(model, states, token, args, device):
    states = [s.clone() for s in states]
    token = token.clone()

    def decode(n):
        nonlocal states, token
        position = args.context_length
        for _ in range(n):
            states, logits = bdh_stream_chunk(model, states, token, start_position=position)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            position += 1

    with torch.no_grad():
        for _ in range(4):
            decode(1)
        return time_decode(decode, args.decode_tokens, device)


def eager_step_fn(step_fn, model, states, token, args, device):
    states = [s.clone() for s in states]
    token = token.clone()
    position = torch.tensor(float(args.context_length), device=device)

    def decode(n):
        nonlocal position
        for _ in range(n):
            logits = step_fn(model, states, token, position)
            token.copy_(torch.argmax(logits[:, -1, :], dim=-1, keepdim=True))
            position = position + 1

    with torch.no_grad():
        for _ in range(4):
            decode(1)
        return time_decode(decode, args.decode_tokens, device)


def manual_graph(step_fn, model, states, token, args, device):
    states = [s.clone() for s in states]
    static_token = token.clone()
    static_position = torch.tensor(float(args.context_length), device=device)

    side_stream = torch.cuda.Stream()
    with torch.no_grad():
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                step_fn(model, states, static_token, static_position)
        torch.cuda.current_stream().wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits = step_fn(model, states, static_token, static_position)

        def decode(n):
            pos = float(args.context_length)
            for _ in range(n):
                static_position.fill_(pos)
                graph.replay()
                static_token.copy_(torch.argmax(static_logits[:, -1, :], dim=-1, keepdim=True))
                pos += 1

        for _ in range(4):
            decode(1)
        return time_decode(decode, args.decode_tokens, device)


def main(args):
    device = torch.device("cuda")
    config = BDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size, dropout=0.0,
    )

    torch.manual_seed(7)
    oracle = BDH(config).to(device=device, dtype=torch.bfloat16).eval()
    torch.manual_seed(7)
    packed_model = PackedEncoderBDH(config).to(device=device, dtype=torch.bfloat16).eval()

    prompt = torch.randint(args.vocab_size, (1, args.context_length), device=device)

    with torch.no_grad():
        base_states = init_bdh_states(oracle, prompt.shape[0], device=device, dtype=torch.bfloat16)
        base_states, logits = bdh_stream_prefill_chunked(oracle, prompt, chunk_length=args.prefill_chunk_length, states=base_states)
        base_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    eager_oracle_tps = eager_baseline(oracle, base_states, base_token, args, device)
    eager_packed_tps = eager_step_fn(bdh_stream_decode_step_packed_encoder_inplace, packed_model, base_states, base_token, args, device)
    graph_oracle_tps = manual_graph(bdh_stream_decode_step_graph_safe_inplace, oracle, base_states, base_token, args, device)
    graph_packed_tps = manual_graph(bdh_stream_decode_step_packed_encoder_inplace, packed_model, base_states, base_token, args, device)

    print(f"context_length={args.context_length}")
    print(f"eager oracle (broadcast encoder):        {eager_oracle_tps:.1f} tok/s")
    print(f"eager packed-encoder:                    {eager_packed_tps:.1f} tok/s")
    print(f"manual-cuda-graph oracle (broadcast):     {graph_oracle_tps:.1f} tok/s")
    print(f"manual-cuda-graph packed-encoder:         {graph_packed_tps:.1f} tok/s")
    print(f"packed/oracle speedup, eager:              {eager_packed_tps/eager_oracle_tps:.3f}x")
    print(f"packed/oracle speedup, cuda-graph:         {graph_packed_tps/graph_oracle_tps:.3f}x")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--prefill-chunk-length", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
