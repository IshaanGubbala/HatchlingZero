#!/usr/bin/env python3
"""Does torch.compile(mode='reduce-overhead') fix the launch-overhead-bound
decode loop? The profile (scripts/hz0h_inference_decode_profile.py) found
370 GPU-kernel launches per single-token Transformer decode step, ~67% of
GPU time in non-matmul overhead (RoPE recompute, LayerNorm stats, KV-cache
concat) -- reduce-overhead mode uses CUDA graphs internally, PyTorch's own
answer to exactly this "many small repeated ops" regime. Tests both
Transformer (KV-cache) and BDH (streaming-state) single-token decode.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM
from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_chunk, bdh_stream_prefill_chunked, init_bdh_states


def time_decode(fn, n, device):
    torch.cuda.synchronize()
    started = time.perf_counter()
    fn(n)
    torch.cuda.synchronize()
    return n / (time.perf_counter() - started)


def test_transformer(args, device):
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

        def eager_decode(n):
            nonlocal token
            for _ in range(n):
                logits = model(token, kv_cache=cache)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        for _ in range(4):
            eager_decode(1)
        eager_tps = time_decode(eager_decode, args.decode_tokens, device)

        compiled_model = torch.compile(model, mode="reduce-overhead")
        cache2 = model.new_kv_cache()
        logits = compiled_model(prompt, kv_cache=cache2)
        token2 = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        def compiled_decode(n):
            nonlocal token2
            for _ in range(n):
                torch.compiler.cudagraph_mark_step_begin()
                logits = compiled_model(token2, kv_cache=cache2)
                token2 = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True).clone()

        for _ in range(4):
            compiled_decode(1)
        compiled_tps = time_decode(compiled_decode, args.decode_tokens, device)

    print(f"transformer eager decode: {eager_tps:.1f} tok/s")
    print(f"transformer compiled(reduce-overhead) decode: {compiled_tps:.1f} tok/s")
    print(f"speedup: {compiled_tps/eager_tps:.2f}x")


def main(args):
    device = torch.device("cuda")
    test_transformer(args, device)


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
