#!/usr/bin/env python3
"""Real decode-throughput and state-memory benchmark: does stacking INT8
base+delta synaptic state (reference/hz0h_bdh_vb_subspace_decoder_int8_state_torch.py,
a real prior-session win -- 32x state reduction, 0% quality loss on plain
VB -- that was never revisited once frozen-identity superseded trainable
VB, and never tried combined with the subspace decoder at all) on top of
the compound BDHVBSubspaceDecoder actually help, or does it repeat the
OLD per-chunk-INT8 regression (a real, disclosed decode-speed hit at
long context that the base+delta design was specifically built to fix
by amortizing quantize/dequantize over merge_every_k tokens instead of
paying it every chunk)?

Same real O(1)-state streaming methodology as every decode benchmark
tonight, including the torch.cuda.empty_cache()-before-timed-region fix
from the earlier context-independence investigation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_vb_subspace_decoder_int8_state_torch import (
    bdh_vb_subspace_decoder_stream_chunk_int8_base_delta_state,
    bdh_vb_subspace_decoder_stream_prefill_chunked_int8_base_delta,
    init_bdh_vb_states_int8_base_delta,
)
from reference.hz0h_bdh_vb_subspace_decoder_stream_torch import bdh_vb_subspace_decoder_stream_chunk, bdh_vb_subspace_decoder_stream_prefill_chunked
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from reference.hz0h_bdh_vb_torch import init_bdh_vb_states
from scripts.hz0h_inference_benchmark import _PowerSampler, _sync, peak_memory_bytes


def measure_plain(model, prompt, max_new_tokens, device, prefill_chunk_length):
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


def measure_int8(model, prompt, max_new_tokens, device, prefill_chunk_length, merge_every_k):
    with torch.no_grad():
        def prefill():
            states = init_bdh_vb_states_int8_base_delta(model, prompt.shape[0], device=device)
            states, logits = bdh_vb_subspace_decoder_stream_prefill_chunked_int8_base_delta(
                model, prompt, chunk_length=prefill_chunk_length, merge_every_k=merge_every_k, states=states)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            return states, token

        def decode(states, token, n_tokens):
            position = prompt.shape[1]
            for _ in range(n_tokens):
                states, logits = bdh_vb_subspace_decoder_stream_chunk_int8_base_delta_state(
                    model, states, token, start_position=position, merge_every_k=merge_every_k)
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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--prefill-chunk-length", type=int, default=2048)
    parser.add_argument("--merge-every-k-values", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from scripts.hz0h_bdh_width_flop_frontier_local import pick_device
    device = pick_device(args.device)

    torch.manual_seed(args.seed)
    config = BDHVBSubspaceDecoderConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                                         mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
                                         d_state=args.d_state, subspace_rank=args.subspace_rank)
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.bfloat16).eval()

    N = args.n_embd * args.mult // args.n_head
    plain_state_bytes_per_layer = args.n_head * N * args.d_state * 2  # bf16, matches model dtype
    int8_base_bytes_per_layer = args.n_head * N * args.d_state * 1  # int8 base, worst case (delta adds on top transiently)
    int8_delta_worst_bytes_per_layer = args.n_head * N * args.d_state * 2  # bf16 delta, right before a merge

    results = {"device": str(device), "dtype": "bfloat16",
               "note": "real decode tok/s + peak memory, plain bf16 VB state vs INT8 base+delta state, compound BDHVBSubspaceDecoder, sweeping merge_every_k (the real amortization knob this design exists to tune).",
               "config": vars(args) | {"out": str(args.out)},
               "analytic_state_bytes_per_layer": {
                   "plain_bf16": plain_state_bytes_per_layer,
                   "int8_base_only": int8_base_bytes_per_layer,
                   "int8_base_plus_worst_case_delta": int8_base_bytes_per_layer + int8_delta_worst_bytes_per_layer,
                   "reduction_factor_vs_plain_worst_case": plain_state_bytes_per_layer / (int8_base_bytes_per_layer + int8_delta_worst_bytes_per_layer),
                   "reduction_factor_vs_plain_base_only": plain_state_bytes_per_layer / int8_base_bytes_per_layer,
               },
               "runs": {}}

    torch.manual_seed(args.seed)
    prompt = torch.randint(0, 256, (1, args.context_length), device=device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    plain = measure_plain(model, prompt, args.decode_tokens, device, args.prefill_chunk_length)
    plain["peak_memory_bytes"] = peak_memory_bytes(device)
    results["runs"]["plain_bf16_state"] = plain
    mem_str = f"{plain['peak_memory_bytes']/1e9:.2f}GB" if plain["peak_memory_bytes"] is not None else "n/a"
    print(f"[plain] {plain['tokens_per_second']:.1f} tok/s peak_mem={mem_str}", flush=True)

    for k in args.merge_every_k_values:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        r = measure_int8(model, prompt, args.decode_tokens, device, args.prefill_chunk_length, merge_every_k=k)
        r["peak_memory_bytes"] = peak_memory_bytes(device)
        r["speedup_vs_plain"] = r["tokens_per_second"] / plain["tokens_per_second"]
        results["runs"][f"int8_base_delta_merge{k}"] = r
        mem_str = f"{r['peak_memory_bytes']/1e9:.2f}GB" if r["peak_memory_bytes"] is not None else "n/a"
        print(f"[int8 merge_every_k={k}] {r['tokens_per_second']:.1f} tok/s "
              f"({r['speedup_vs_plain']:.3f}x vs plain) peak_mem={mem_str}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
