#!/usr/bin/env python3
"""Real batch>1 decode-throughput and batch-size-frontier check for the
compound VB-frozen-identity + subspace-decoder architecture, one of the
untested follow-ups flagged after the (now-fixed) single-batch decode
benchmarks (results/local/hz0h_vb_subspace_decoder_decode_benchmark.json,
flat 2.37x compound speedup at batch=1). Everything measured so far this
session used batch=1 -- this sweeps real batch sizes (doubling until
OOM) at a fixed moderate context, for exact BDH and the compound model,
same real O(1)-state streaming decode path as every other decode
benchmark tonight (includes the torch.cuda.empty_cache()-before-timed-
region fix from the context-independence investigation).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_chunk, bdh_stream_prefill_chunked, init_bdh_states
from reference.hz0h_bdh_vb_subspace_decoder_stream_torch import bdh_vb_subspace_decoder_stream_chunk, bdh_vb_subspace_decoder_stream_prefill_chunked
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from reference.hz0h_bdh_vb_torch import init_bdh_vb_states
from scripts.hz0h_inference_benchmark import _PowerSampler, _sync, peak_memory_bytes, reset_peak_memory


def _measure(prefill_fn, decode_step_fn, init_states_fn, model, prompt, max_new_tokens, device, prefill_chunk_length):
    with torch.no_grad():
        def prefill():
            states = init_states_fn(model, prompt.shape[0], device=device)
            states, logits = prefill_fn(model, prompt, chunk_length=prefill_chunk_length, states=states)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            return states, token

        def decode(states, token, n_tokens):
            position = prompt.shape[1]
            for _ in range(n_tokens):
                states, logits = decode_step_fn(model, states, token, start_position=position)
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
    total_tokens = max_new_tokens * prompt.shape[0]
    return {"aggregate_tokens_per_second": total_tokens / elapsed, "per_sequence_tokens_per_second": max_new_tokens / elapsed,
            "elapsed_seconds": elapsed, "mean_watts": sampler.mean_watts(), "batch_size": prompt.shape[0]}


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
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--prefill-chunk-length", type=int, default=2048)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from scripts.hz0h_bdh_width_flop_frontier_local import pick_device
    device = pick_device(args.device)

    torch.manual_seed(args.seed)
    bdh_config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                            mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0)
    bdh_model = BDH(bdh_config).to(device=device, dtype=torch.bfloat16).eval()

    torch.manual_seed(args.seed)
    compound_config = BDHVBSubspaceDecoderConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                                                  mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
                                                  d_state=args.d_state, subspace_rank=args.subspace_rank)
    compound_model = BDHVBSubspaceDecoder(compound_config).to(device=device, dtype=torch.bfloat16).eval()

    results = {"device": str(device), "dtype": "bfloat16",
               "note": "untrained execution-speed diagnostic. batch>1 real streaming decode + batch-size-frontier sweep (doubling until OOM), fixed context, for exact BDH vs the compound (VB d_state=624 + subspace r=64) architecture.",
               "config": {"n_embd": args.n_embd, "mult": args.mult, "n_layer": args.n_layer, "n_head": args.n_head,
                          "d_state": args.d_state, "subspace_rank": args.subspace_rank, "context_length": args.context_length,
                          "decode_tokens": args.decode_tokens, "seed": args.seed},
               "by_batch_size": {}}

    bdh_max_batch = None
    compound_max_batch = None

    for batch_size in args.batch_sizes:
        entry = {}
        torch.manual_seed(args.seed)
        prompt = torch.randint(0, 256, (batch_size, args.context_length), device=device)

        if bdh_max_batch is None:
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    reset_peak_memory(device)  # real fix, 2026-08-26: without this, peak_memory_bytes() (torch.cuda.max_memory_allocated()) is a running max NEVER reset across the whole process -- every batch size's "peak" was contaminated by whatever the highest point was in an EARLIER iteration, not that batch size's own peak. Caught by a user question about a 23.27GB reading that was ~15.7x the analytic persistent-state size.
                bdh_r = _measure(bdh_stream_prefill_chunked, bdh_stream_chunk, init_bdh_states,
                                  bdh_model, prompt, args.decode_tokens, device, args.prefill_chunk_length)
                bdh_r["peak_memory_bytes"] = peak_memory_bytes(device)
                entry["bdh"] = bdh_r
            except torch.cuda.OutOfMemoryError as exc:
                entry["bdh"] = {"status": "OOM", "detail": str(exc)[:200]}
                bdh_max_batch = args.batch_sizes[args.batch_sizes.index(batch_size) - 1] if args.batch_sizes.index(batch_size) > 0 else 0
                torch.cuda.empty_cache()

        if compound_max_batch is None:
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    reset_peak_memory(device)
                compound_r = _measure(bdh_vb_subspace_decoder_stream_prefill_chunked, bdh_vb_subspace_decoder_stream_chunk, init_bdh_vb_states,
                                       compound_model, prompt, args.decode_tokens, device, args.prefill_chunk_length)
                compound_r["peak_memory_bytes"] = peak_memory_bytes(device)
                entry["compound"] = compound_r
            except torch.cuda.OutOfMemoryError as exc:
                entry["compound"] = {"status": "OOM", "detail": str(exc)[:200]}
                compound_max_batch = args.batch_sizes[args.batch_sizes.index(batch_size) - 1] if args.batch_sizes.index(batch_size) > 0 else 0
                torch.cuda.empty_cache()

        if "bdh" in entry and "compound" in entry and "aggregate_tokens_per_second" in entry.get("bdh", {}) and "aggregate_tokens_per_second" in entry.get("compound", {}):
            entry["compound_over_bdh_speedup"] = entry["compound"]["aggregate_tokens_per_second"] / entry["bdh"]["aggregate_tokens_per_second"]
            print(f"[batch={batch_size}] BDH {entry['bdh']['aggregate_tokens_per_second']:.1f} tok/s agg | "
                  f"Compound {entry['compound']['aggregate_tokens_per_second']:.1f} tok/s agg | "
                  f"speedup={entry['compound_over_bdh_speedup']:.3f}x", flush=True)
        else:
            print(f"[batch={batch_size}] bdh={'OOM' if 'status' in entry.get('bdh', {}) else 'ok'} "
                  f"compound={'OOM' if 'status' in entry.get('compound', {}) else 'ok'}", flush=True)

        results["by_batch_size"][str(batch_size)] = entry
        if bdh_max_batch is not None and compound_max_batch is not None:
            break

    results["bdh_max_working_batch_size"] = bdh_max_batch
    results["compound_max_working_batch_size"] = compound_max_batch

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
