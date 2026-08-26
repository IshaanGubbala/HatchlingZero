#!/usr/bin/env python3
"""Tier 0 of plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md:
a production-style (preallocated, no per-token torch.cat) Transformer
decode baseline, re-run against BDH's already-established streaming
decode path on the SAME GPU/config -- the missing fairness fix the plan
calls mandatory before trusting the earlier BDH-vs-Transformer long-
context decode crossover.

Reuses BDH's real streaming decode measurement
(measure_bdh_prefill/measure_bdh_decode_streaming,
scripts/hz0h_inference_benchmark.py) unmodified; the only new mechanism
is the Transformer side (reference/hz0h_matched_transformer_static_kv.py),
verified bit-exact against the existing cat-based KV-cache path before
this script's numbers are trusted.
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
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_matched_transformer_static_kv import StaticKVMatchedTransformerLM
from scripts.hz0h_inference_benchmark import (
    _PowerSampler,
    _sync,
    measure_bdh_decode_streaming,
    measure_bdh_prefill,
    measure_transformer_prefill,
    peak_memory_bytes,
    resolve_device,
    reset_peak_memory,
)


def measure_transformer_decode_static_kv(matched_model: MatchedTransformerLM, prompt: torch.Tensor, max_new_tokens: int, device: torch.device, max_seq_len: int) -> dict:
    """The real, no-per-token-realloc decode path: preallocate the KV
    buffer ONCE to max_seq_len (prompt length + max_new_tokens), prefill
    once outside the timed region, then decode one token at a time doing
    only an in-place slice write per step (StaticKVCache.write) instead
    of the cat-based path's full-tensor reallocation+copy every step."""
    static_model = StaticKVMatchedTransformerLM(matched_model).eval()
    with torch.no_grad():
        def prefill() -> tuple:
            cache = static_model.new_cache(batch_size=prompt.shape[0], max_seq_len=max_seq_len, device=device, dtype=next(matched_model.parameters()).dtype)
            logits = static_model(prompt, cache)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            return cache, token

        def decode(cache, token: torch.Tensor, n_tokens: int) -> None:
            for _ in range(n_tokens):
                logits = static_model(token, cache)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        _sync(device)
        cache, token = prefill()
        decode(cache, token, min(4, max_new_tokens))  # warmup
        _sync(device)

        cache, token = prefill()
        _sync(device)
        with _PowerSampler(device) as sampler:
            started = time.perf_counter()
            decode(cache, token, max_new_tokens)
            _sync(device)
            elapsed = time.perf_counter() - started
    return {"tokens_per_second": max_new_tokens / elapsed, "elapsed_seconds": elapsed, "mean_watts": sampler.mean_watts()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=3, help="Production-scale matched-param Transformer used 3 layers at d_model=2496 in the earlier session's crossover test.")
    parser.add_argument("--transformer-heads", type=int, default=None)
    parser.add_argument("--transformer-head-dim", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None, help="Defaults to n_embd * mlp_internal_dim_multiplier // n_head (matches BDH's own N-per-head width) if not set.")
    parser.add_argument("--context-lengths", type=str, default="128,2048,16384,65536")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--prefill-repeats", type=int, default=5)
    parser.add_argument("--prefill-chunk-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    bdh_config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head, mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size, dropout=0.0)
    bdh_model = BDH(bdh_config).to(device=device, dtype=torch_dtype)
    bdh_model.attn.freqs = bdh_model.attn.freqs.to(torch.float32)
    bdh_model.eval()

    transformer_heads = args.transformer_heads if args.transformer_heads is not None else args.n_head
    transformer_head_dim = args.transformer_head_dim if args.transformer_head_dim is not None else args.n_embd // transformer_heads
    d_ff = args.d_ff if args.d_ff is not None else args.n_embd * args.mlp_internal_dim_multiplier // args.n_head
    transformer_config = MatchedTransformerConfig({
        "vocab_size": args.vocab_size, "d_model": args.n_embd, "num_layers": args.transformer_layers,
        "num_heads": transformer_heads, "head_dim": transformer_head_dim, "d_ff": d_ff, "use_rope": True,
    })
    transformer_model = MatchedTransformerLM(transformer_config).to(device=device, dtype=torch_dtype)
    transformer_model.eval()

    bdh_params = sum(p.numel() for p in bdh_model.parameters())
    transformer_params = sum(p.numel() for p in transformer_model.parameters())

    results: dict = {
        "device": str(device), "dtype": args.dtype,
        "bdh_parameter_count": bdh_params, "transformer_parameter_count": transformer_params,
        "trained_weights": False, "note": "untrained execution-speed diagnostic -- decode tok/s and memory only, not a quality claim",
        "decode_tokens": args.decode_tokens, "seed": args.seed,
        "by_context_length": {},
    }

    for context_length in (int(x) for x in args.context_lengths.split(",") if x.strip()):
        prompt = torch.randint(0, args.vocab_size, (1, context_length), device=device)
        max_seq_len = context_length + args.decode_tokens + 8

        reset_peak_memory(device)
        bdh_prefill = measure_bdh_prefill(bdh_model, prompt, args.prefill_repeats, device, args.prefill_chunk_length)
        bdh_prefill["peak_memory_bytes"] = peak_memory_bytes(device)

        reset_peak_memory(device)
        bdh_decode_streaming = measure_bdh_decode_streaming(bdh_model, prompt, args.decode_tokens, device, args.prefill_chunk_length)
        bdh_decode_streaming["peak_memory_bytes"] = peak_memory_bytes(device)

        reset_peak_memory(device)
        transformer_prefill = measure_transformer_prefill(transformer_model, prompt, args.prefill_repeats, device)
        transformer_prefill["peak_memory_bytes"] = peak_memory_bytes(device)

        reset_peak_memory(device)
        transformer_decode_static_kv = measure_transformer_decode_static_kv(transformer_model, prompt, args.decode_tokens, device, max_seq_len)
        transformer_decode_static_kv["peak_memory_bytes"] = peak_memory_bytes(device)

        results["by_context_length"][context_length] = {
            "bdh_prefill": bdh_prefill,
            "bdh_decode_streaming": bdh_decode_streaming,
            "transformer_prefill": transformer_prefill,
            "transformer_decode_static_kv": transformer_decode_static_kv,
            "bdh_decode_vs_transformer_decode_speedup": bdh_decode_streaming["tokens_per_second"] / transformer_decode_static_kv["tokens_per_second"],
        }
        print(f"[context={context_length}] BDH decode {bdh_decode_streaming['tokens_per_second']:.1f} tok/s | "
              f"Transformer static-KV decode {transformer_decode_static_kv['tokens_per_second']:.1f} tok/s | "
              f"BDH/Transformer = {results['by_context_length'][context_length]['bdh_decode_vs_transformer_decode_speedup']:.3f}x", flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[done] wrote {args.output}", flush=True)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
