#!/usr/bin/env python3
"""Diagnostic: does hz0h_inference_benchmark.py's background nvidia-smi
power-polling thread (spawning a subprocess every 50ms DURING the timed
decode region) contaminate the throughput numbers it reports?

Real 300M-param decode throughput of ~40-360 tok/s on an A40 looked
suspiciously low -- this reruns the exact same BDH streaming-state and
Transformer KV-cache decode loops at context=512, with and without the
_PowerSampler wrapper active, to isolate whether the measurement
methodology itself is the bottleneck.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM
from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_chunk, bdh_stream_prefill_chunked, init_bdh_states
from scripts.hz0h_inference_benchmark import _PowerSampler, _sync


def transformer_decode(model, prompt, n_tokens, device, use_power_sampler):
    with torch.no_grad():
        def prefill():
            cache = model.new_kv_cache()
            logits = model(prompt, kv_cache=cache)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            return cache, token

        def decode(cache, token, n):
            for _ in range(n):
                logits = model(token, kv_cache=cache)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        _sync(device)
        cache, token = prefill()
        decode(cache, token, 4)
        _sync(device)

        cache, token = prefill()
        _sync(device)
        if use_power_sampler:
            with _PowerSampler(device):
                started = time.perf_counter()
                decode(cache, token, n_tokens)
                _sync(device)
                elapsed = time.perf_counter() - started
        else:
            started = time.perf_counter()
            decode(cache, token, n_tokens)
            _sync(device)
            elapsed = time.perf_counter() - started
    return n_tokens / elapsed


def bdh_streaming_decode(model, prompt, n_tokens, device, prefill_chunk_length, use_power_sampler):
    with torch.no_grad():
        def prefill():
            states = init_bdh_states(model, prompt.shape[0], device=device)
            states, logits = bdh_stream_prefill_chunked(model, prompt, chunk_length=prefill_chunk_length, states=states)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            return states, token

        def decode(states, token, n):
            position = prompt.shape[1]
            for _ in range(n):
                states, logits = bdh_stream_chunk(model, states, token, start_position=position)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                position += 1

        _sync(device)
        states, token = prefill()
        decode(states, token, 4)
        _sync(device)

        states, token = prefill()
        _sync(device)
        if use_power_sampler:
            with _PowerSampler(device):
                started = time.perf_counter()
                decode(states, token, n_tokens)
                _sync(device)
                elapsed = time.perf_counter() - started
        else:
            started = time.perf_counter()
            decode(states, token, n_tokens)
            _sync(device)
            elapsed = time.perf_counter() - started
    return n_tokens / elapsed


def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    bdh_config = BDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size, dropout=0.0,
    )
    bdh_model = BDH(bdh_config).to(device=device, dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)
    bdh_model.eval()

    transformer_config = MatchedTransformerConfig({
        "vocab_size": args.vocab_size, "d_model": args.n_embd, "num_layers": args.transformer_layers,
        "num_heads": args.transformer_heads, "head_dim": args.n_embd // args.transformer_heads,
        "d_ff": args.d_ff, "use_rope": True,
    })
    transformer_model = MatchedTransformerLM(transformer_config).to(device=device, dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)
    transformer_model.eval()

    prompt = torch.randint(args.vocab_size, (1, args.context_length), device=device)

    report = {
        "context_length": args.context_length,
        "decode_tokens": args.decode_tokens,
        "transformer_with_power_sampler": transformer_decode(transformer_model, prompt, args.decode_tokens, device, True),
        "transformer_without_power_sampler": transformer_decode(transformer_model, prompt, args.decode_tokens, device, False),
        "bdh_streaming_with_power_sampler": bdh_streaming_decode(bdh_model, prompt, args.decode_tokens, device, args.prefill_chunk_length, True),
        "bdh_streaming_without_power_sampler": bdh_streaming_decode(bdh_model, prompt, args.decode_tokens, device, args.prefill_chunk_length, False),
    }
    report["transformer_slowdown_from_power_sampler"] = report["transformer_without_power_sampler"] / report["transformer_with_power_sampler"]
    report["bdh_slowdown_from_power_sampler"] = report["bdh_streaming_without_power_sampler"] / report["bdh_streaming_with_power_sampler"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=16)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=9984)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--prefill-chunk-length", type=int, default=1024)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
