#!/usr/bin/env python3
"""The real next step flagged after the compound architecture's own
decode benchmarks: everything measured for BDHVBSubspaceDecoder so far
was BDH-vs-BDH (compound vs exact-BDH baseline). This project's actual
central thesis -- does BDH's structural difference from a Transformer
translate into a measurable advantage -- was last tested with exact BDH
against the fair static-KV Transformer (Tier 0 item 3,
scripts/hz0h_transformer_static_kv_decode_benchmark.py): Transformer won
decisively at context<=16384 (3.36x-4.83x), BDH only near-tied
(1.03x) near context=65536. That comparison has never been rerun with
the compound model, which decodes 2.25-2.30x faster than exact BDH
alone -- this script is the real, direct answer, not a back-of-envelope
projection from the two numbers separately.

Reuses reference/hz0h_matched_transformer_static_kv.py (the same fair,
preallocated-KV Transformer baseline, already bit-exact-verified against
the cat-based path) unmodified. Real O(1)-state streaming decode for the
compound model via reference/hz0h_bdh_vb_subspace_decoder_stream_torch.py,
with the torch.cuda.empty_cache()-before-timed-region fix from the
earlier context-independence investigation (both sides, for symmetry).
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
from reference.hz0h_bdh_vb_subspace_decoder_stream_torch import bdh_vb_subspace_decoder_stream_chunk, bdh_vb_subspace_decoder_stream_prefill_chunked
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from reference.hz0h_bdh_vb_torch import init_bdh_vb_states
from reference.hz0h_matched_transformer_static_kv import StaticKVMatchedTransformerLM
from scripts.hz0h_inference_benchmark import _PowerSampler, _sync, peak_memory_bytes


def measure_transformer_decode_static_kv(matched_model, prompt, max_new_tokens, device, max_seq_len):
    static_model = StaticKVMatchedTransformerLM(matched_model).eval()
    with torch.no_grad():
        def prefill():
            cache = static_model.new_cache(batch_size=prompt.shape[0], max_seq_len=max_seq_len, device=device, dtype=next(matched_model.parameters()).dtype)
            logits = static_model(prompt, cache)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            return cache, token

        def decode(cache, token, n_tokens):
            for _ in range(n_tokens):
                logits = static_model(token, cache)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        _sync(device)
        cache, token = prefill()
        decode(cache, token, min(4, max_new_tokens))
        _sync(device)

        cache, token = prefill()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _sync(device)
        with _PowerSampler(device) as sampler:
            started = time.perf_counter()
            decode(cache, token, max_new_tokens)
            _sync(device)
            elapsed = time.perf_counter() - started
    return {"tokens_per_second": max_new_tokens / elapsed, "elapsed_seconds": elapsed, "mean_watts": sampler.mean_watts()}


def measure_bdh_decode(model, prompt, max_new_tokens, device, prefill_chunk_length):
    with torch.no_grad():
        def prefill():
            states = init_bdh_states(model, prompt.shape[0], device=device)
            states, logits = bdh_stream_prefill_chunked(model, prompt, chunk_length=prefill_chunk_length, states=states)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            return states, token

        def decode(states, token, n_tokens):
            position = prompt.shape[1]
            for _ in range(n_tokens):
                states, logits = bdh_stream_chunk(model, states, token, start_position=position)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                position += 1

        _sync(device)
        states, token = prefill()
        decode(states, token, 4)
        _sync(device)
        states, token = prefill()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _sync(device)
        with _PowerSampler(device) as sampler:
            started = time.perf_counter()
            decode(states, token, max_new_tokens)
            _sync(device)
            elapsed = time.perf_counter() - started
    return {"tokens_per_second": max_new_tokens / elapsed, "elapsed_seconds": elapsed, "mean_watts": sampler.mean_watts()}


def measure_compound_decode(model, prompt, max_new_tokens, device, prefill_chunk_length):
    with torch.no_grad():
        def prefill():
            states = init_bdh_vb_states(model, prompt.shape[0], device=device)
            states, logits = bdh_vb_subspace_decoder_stream_prefill_chunked(model, prompt, chunk_length=prefill_chunk_length, states=states)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            return states, token

        def decode(states, token, n_tokens):
            position = prompt.shape[1]
            for _ in range(n_tokens):
                states, logits = bdh_vb_subspace_decoder_stream_chunk(model, states, token, start_position=position)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                position += 1

        _sync(device)
        states, token = prefill()
        decode(states, token, 4)
        _sync(device)
        states, token = prefill()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _sync(device)
        with _PowerSampler(device) as sampler:
            started = time.perf_counter()
            decode(states, token, max_new_tokens)
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
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--transformer-heads", type=int, default=None)
    parser.add_argument("--transformer-head-dim", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--context-lengths", type=str, default="128,2048,16384,65536")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--prefill-chunk-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from scripts.hz0h_bdh_width_flop_frontier_local import pick_device
    device = pick_device(args.device)
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    torch.manual_seed(args.seed)

    bdh_config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                            mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size, dropout=0.0)
    bdh_model = BDH(bdh_config).to(device=device, dtype=torch_dtype)
    bdh_model.attn.freqs = bdh_model.attn.freqs.to(torch.float32)
    bdh_model.eval()

    torch.manual_seed(args.seed)
    compound_config = BDHVBSubspaceDecoderConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                                                  mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size, dropout=0.0,
                                                  d_state=args.d_state, subspace_rank=args.subspace_rank)
    compound_model = BDHVBSubspaceDecoder(compound_config).to(device=device, dtype=torch_dtype)
    compound_model.attn.freqs = compound_model.attn.freqs.to(torch.float32)
    compound_model.eval()

    transformer_heads = args.transformer_heads if args.transformer_heads is not None else args.n_head
    transformer_head_dim = args.transformer_head_dim if args.transformer_head_dim is not None else args.n_embd // transformer_heads
    d_ff = args.d_ff if args.d_ff is not None else args.n_embd * args.mlp_internal_dim_multiplier // args.n_head
    transformer_config = MatchedTransformerConfig({
        "vocab_size": args.vocab_size, "d_model": args.n_embd, "num_layers": args.transformer_layers,
        "num_heads": transformer_heads, "head_dim": transformer_head_dim, "d_ff": d_ff, "use_rope": True,
    })
    transformer_model = MatchedTransformerLM(transformer_config).to(device=device, dtype=torch_dtype)
    transformer_model.eval()

    results = {
        "device": str(device), "dtype": args.dtype,
        "bdh_parameter_count": sum(p.numel() for p in bdh_model.parameters()),
        "compound_parameter_count": sum(p.numel() for p in compound_model.parameters()),
        "transformer_parameter_count": sum(p.numel() for p in transformer_model.parameters()),
        "trained_weights": False,
        "note": "untrained execution-speed diagnostic -- decode tok/s only, not a quality claim (quality already established separately). Real head-to-head this project's central thesis needed: compound BDH (state-compressed + subspace-decoder) vs the SAME fair static-KV Transformer baseline used for Tier 0's original exact-BDH-vs-Transformer comparison.",
        "decode_tokens": args.decode_tokens, "seed": args.seed,
        "by_context_length": {},
    }

    for context_length in (int(x) for x in args.context_lengths.split(",") if x.strip()):
        torch.manual_seed(args.seed)
        prompt = torch.randint(0, args.vocab_size, (1, context_length), device=device)
        max_seq_len = context_length + args.decode_tokens + 8
        try:
            bdh_decode = measure_bdh_decode(bdh_model, prompt, args.decode_tokens, device, args.prefill_chunk_length)
            bdh_decode["peak_memory_bytes"] = peak_memory_bytes(device)

            compound_decode = measure_compound_decode(compound_model, prompt, args.decode_tokens, device, args.prefill_chunk_length)
            compound_decode["peak_memory_bytes"] = peak_memory_bytes(device)

            transformer_decode = measure_transformer_decode_static_kv(transformer_model, prompt, args.decode_tokens, device, max_seq_len)
            transformer_decode["peak_memory_bytes"] = peak_memory_bytes(device)

            entry = {
                "bdh_decode_streaming": bdh_decode,
                "compound_decode_streaming": compound_decode,
                "transformer_decode_static_kv": transformer_decode,
                "bdh_over_transformer_speedup": bdh_decode["tokens_per_second"] / transformer_decode["tokens_per_second"],
                "compound_over_transformer_speedup": compound_decode["tokens_per_second"] / transformer_decode["tokens_per_second"],
            }
            results["by_context_length"][context_length] = entry
            print(f"[context={context_length}] BDH {bdh_decode['tokens_per_second']:.1f} tok/s "
                  f"({entry['bdh_over_transformer_speedup']:.3f}x vs Transformer) | "
                  f"Compound {compound_decode['tokens_per_second']:.1f} tok/s "
                  f"({entry['compound_over_transformer_speedup']:.3f}x vs Transformer) | "
                  f"Transformer {transformer_decode['tokens_per_second']:.1f} tok/s", flush=True)
        except torch.cuda.OutOfMemoryError as exc:
            results["by_context_length"][context_length] = {"status": "OOM", "detail": str(exc)[:300]}
            print(f"[context={context_length}] OOM: {exc}", flush=True)
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
