#!/usr/bin/env python3
"""Real decode-throughput benchmark for Tier 4 item 23 (Subspace BDH,
SVD-warmstarted rank-64 decoder). Every result gathered so far for this
architecture (results/local/hz0h_subspace_decoder_warmstart_r64*.json)
is a TRAINING-time comparison (wall-clock seconds for a fixed token
budget). This repo's own /goal is training AND inference throughput --
this script measures the piece that was still missing: real batch=1
streaming decode tok/s, matching the exact methodology already
established for the fair BDH-vs-Transformer comparison
(scripts/hz0h_inference_benchmark.py's measure_bdh_decode_streaming,
scripts/hz0h_transformer_static_kv_decode_benchmark.py) -- real O(1)-state
streaming path (bdh_stream_chunk / bdh_subspace_decoder_stream_chunk),
not a naive full-replay generate() loop.

The decoder is weight-tied and reused every one of `n_layer` rounds per
generated token, and (being a plain dense nn.Parameter, not something
the framework caches across calls) gets re-read from HBM every round in
the streaming decode path -- so shrinking it from
nh*N*D (~99.7M params, ~199MB in bf16) to
nh*N*r + r*D (~2.7M params, ~5.4MB in bf16 at r=64) is a real per-round
HBM traffic cut on a path Tier 0 of the plan already established is
memory-bandwidth-bound, not compute-bound -- exactly where this kind of
compression should show up as real decode speedup, not just a training
compute-time win.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_subspace_decoder_torch import BDHSubspaceDecoder, BDHSubspaceDecoderConfig
from reference.hz0h_bdh_subspace_decoder_stream_torch import bdh_subspace_decoder_stream_chunk, bdh_subspace_decoder_stream_prefill_chunked
from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_chunk, bdh_stream_prefill_chunked, init_bdh_states
from scripts.hz0h_inference_benchmark import _PowerSampler, _sync, peak_memory_bytes


def measure_subspace_decode_streaming(model: BDHSubspaceDecoder, prompt: torch.Tensor, max_new_tokens: int, device: torch.device, prefill_chunk_length: int) -> dict:
    with torch.no_grad():
        def prefill():
            states = init_bdh_states(model, prompt.shape[0], device=device)
            states, logits = bdh_subspace_decoder_stream_prefill_chunked(model, prompt, chunk_length=prefill_chunk_length, states=states)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            return states, token

        def decode(states, token, n_tokens):
            position = prompt.shape[1]
            for _ in range(n_tokens):
                states, logits = bdh_subspace_decoder_stream_chunk(model, states, token, start_position=position)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                position += 1

        _sync(device)
        states, token = prefill()
        decode(states, token, 4)  # warmup
        _sync(device)

        states, token = prefill()
        if device.type == "cuda":
            torch.cuda.empty_cache()  # real fix, 2026-08-25: prefill's large chunked buffers fragment the caching allocator; clearing right before the timed region (not inside it) stops that fragmentation from bleeding into decode's own small per-step allocations -- see scripts/hz0h_bdh_decode_context_independence_check.py, which isolated and confirmed this is what was causing decode tok/s to falsely appear context-dependent (56.1-56.5 tok/s flat once fixed, vs a spurious ~2x falloff before)
        _sync(device)
        with _PowerSampler(device) as sampler:
            started = __import__("time").perf_counter()
            decode(states, token, max_new_tokens)
            _sync(device)
            elapsed = __import__("time").perf_counter() - started
    return {"tokens_per_second": max_new_tokens / elapsed, "elapsed_seconds": elapsed, "mean_watts": sampler.mean_watts()}


def measure_bdh_decode_streaming(model: BDH, prompt: torch.Tensor, max_new_tokens: int, device: torch.device, prefill_chunk_length: int) -> dict:
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
        decode(states, token, 4)  # warmup
        _sync(device)

        states, token = prefill()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _sync(device)
        with _PowerSampler(device) as sampler:
            started = __import__("time").perf_counter()
            decode(states, token, max_new_tokens)
            _sync(device)
            elapsed = __import__("time").perf_counter() - started
    return {"tokens_per_second": max_new_tokens / elapsed, "elapsed_seconds": elapsed, "mean_watts": sampler.mean_watts()}


def correctness_check(bdh_model: BDH, subspace_model: BDHSubspaceDecoder, device: torch.device) -> float:
    """Sanity check the two streaming implementations are wired the same
    way (not a numerical-equivalence claim -- the two models have
    DIFFERENT weights -- just confirms both code paths run and produce
    finite, real logits of the expected shape before trusting timing."""
    idx = torch.randint(0, 256, (1, 8), device=device)
    with torch.no_grad():
        _, logits_a = bdh_stream_prefill_chunked(bdh_model, idx, chunk_length=8)
        _, logits_b = bdh_subspace_decoder_stream_prefill_chunked(subspace_model, idx, chunk_length=8)
    assert logits_a.shape == logits_b.shape, (logits_a.shape, logits_b.shape)
    assert torch.isfinite(logits_a).all() and torch.isfinite(logits_b).all()
    return float((logits_a - logits_b).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[128, 2048, 16384])
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--prefill-chunk-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from scripts.hz0h_bdh_width_flop_frontier_local import pick_device
    device = pick_device(args.device)

    torch.manual_seed(args.seed)
    bdh_config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                            mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0)
    bdh_model = BDH(bdh_config).to(device=device, dtype=torch.bfloat16).eval()

    torch.manual_seed(args.seed)
    subspace_config = BDHSubspaceDecoderConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                                                mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
                                                subspace_rank=args.subspace_rank)
    subspace_model = BDHSubspaceDecoder(subspace_config).to(device=device, dtype=torch.bfloat16).eval()

    decoder_dense_bytes = bdh_model.decoder.numel() * bdh_model.decoder.element_size()
    decoder_subspace_bytes = (subspace_model.decoder_up.numel() * subspace_model.decoder_up.element_size()
                               + subspace_model.decoder_down.numel() * subspace_model.decoder_down.element_size())

    diff = correctness_check(bdh_model, subspace_model, device)
    print(f"[sanity] streaming paths both run, both finite, shapes match (informational max diff between DIFFERENT-weight models: {diff:.3f})", flush=True)

    results = {
        "device": str(device), "dtype": "bfloat16",
        "note": "untrained execution-speed diagnostic -- decode tok/s and memory only, not a quality claim (quality already established separately: results/local/hz0h_subspace_decoder_warmstart_r64*.json). Real O(1)-state streaming decode path for both models (bdh_stream_chunk / bdh_subspace_decoder_stream_chunk), not naive full-replay.",
        "config": {"n_embd": args.n_embd, "mult": args.mult, "n_layer": args.n_layer, "n_head": args.n_head,
                    "subspace_rank": args.subspace_rank, "decode_tokens": args.decode_tokens, "seed": args.seed},
        "decoder_weight_bytes": {"dense": decoder_dense_bytes, "subspace": decoder_subspace_bytes,
                                  "reduction_factor": decoder_dense_bytes / decoder_subspace_bytes},
        "by_context_length": {},
    }

    for context_length in args.context_lengths:
        torch.manual_seed(args.seed)
        prompt = torch.randint(0, 256, (1, context_length), device=device)
        try:
            bdh_decode = measure_bdh_decode_streaming(bdh_model, prompt, args.decode_tokens, device, args.prefill_chunk_length)
            bdh_decode["peak_memory_bytes"] = peak_memory_bytes(device)
            subspace_decode = measure_subspace_decode_streaming(subspace_model, prompt, args.decode_tokens, device, args.prefill_chunk_length)
            subspace_decode["peak_memory_bytes"] = peak_memory_bytes(device)
            speedup = subspace_decode["tokens_per_second"] / bdh_decode["tokens_per_second"]
            results["by_context_length"][str(context_length)] = {
                "bdh_decode_streaming": bdh_decode,
                "subspace_decode_streaming": subspace_decode,
                "subspace_over_bdh_decode_speedup": speedup,
            }
            print(f"[context={context_length}] BDH decode {bdh_decode['tokens_per_second']:.1f} tok/s | "
                  f"Subspace decode {subspace_decode['tokens_per_second']:.1f} tok/s | "
                  f"Subspace/BDH = {speedup:.3f}x", flush=True)
        except torch.cuda.OutOfMemoryError as exc:
            results["by_context_length"][str(context_length)] = {"status": "OOM", "detail": str(exc)}
            print(f"[context={context_length}] OOM: {exc}", flush=True)
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
