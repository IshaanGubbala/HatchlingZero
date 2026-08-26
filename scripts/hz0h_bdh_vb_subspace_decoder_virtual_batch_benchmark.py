#!/usr/bin/env python3
"""Phase A of the concurrency-scaling investigation: does physically
batching decode requests together cause the batch-frontier benchmark's
observed throughput collapse (156 -> 75 -> 78 aggregate tok/s at
B=1/2/4), or is that collapse specific to constructing one big B-sized
tensor per step? Tests the alternative: many resident per-request
states (B=1 shape each), round-robined through single-request decode
steps -- "virtual batch N, physical microbatch 1" -- the standard LLM
continuous-batching pattern, here applied without ever constructing a
tensor batched across requests.

If this holds close to the real B=1 aggregate ceiling (156 tok/s) at
N=32 resident requests, the earlier B>1 collapse is a property of the
CURRENT physically-batched execution path, not of the persistent state
itself -- exactly the distinction the user's own math flagged (analytic
state at B=4 is ~1.49GiB, measured peak was 23.27GB, ~15.7x higher).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_vb_subspace_decoder_stream_torch import bdh_vb_subspace_decoder_stream_chunk, bdh_vb_subspace_decoder_stream_prefill_chunked
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from reference.hz0h_bdh_vb_torch import init_bdh_vb_states
from scripts.hz0h_inference_benchmark import _PowerSampler, _sync, peak_memory_bytes


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
    parser.add_argument("--decode-tokens-per-request", type=int, default=16)
    parser.add_argument("--prefill-chunk-length", type=int, default=2048)
    parser.add_argument("--resident-request-counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from scripts.hz0h_bdh_width_flop_frontier_local import pick_device
    device = pick_device(args.device)

    torch.manual_seed(args.seed)
    config = BDHVBSubspaceDecoderConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                                         mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
                                         d_state=args.d_state, subspace_rank=args.subspace_rank)
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.bfloat16).eval()

    results = {"device": str(device), "dtype": "bfloat16",
               "note": "virtual-batch decode: N resident B=1-shaped per-request states, round-robined through single-request decode steps, no physically-batched tensor ever constructed. Compare aggregate tok/s and peak memory against the physically-batched numbers in hz0h_vb_subspace_decoder_batch_frontier.json (B1=156.2, B2=75.3, B4=77.6 aggregate tok/s).",
               "config": vars(args) | {"out": str(args.out)},
               "by_resident_request_count": {}}

    for n_requests in args.resident_request_counts:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Prefill each request independently at B=1 -- never construct a
            # batched prefill tensor, matching the phase's own premise.
            request_states = []
            request_tokens = []
            with torch.no_grad():
                for r in range(n_requests):
                    torch.manual_seed(args.seed + r)
                    prompt = torch.randint(0, 256, (1, args.context_length), device=device)
                    states = init_bdh_vb_states(model, 1, device=device)
                    states, logits = bdh_vb_subspace_decoder_stream_prefill_chunked(
                        model, prompt, chunk_length=args.prefill_chunk_length, states=states)
                    token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    request_states.append(states)
                    request_tokens.append(token)
                    del prompt, logits

            _sync(device)

            def run_round_robin(n_steps_per_request):
                positions = [args.context_length] * n_requests
                with torch.no_grad():
                    for _ in range(n_steps_per_request):
                        for r in range(n_requests):
                            states, logits = bdh_vb_subspace_decoder_stream_chunk(
                                model, request_states[r], request_tokens[r], start_position=positions[r])
                            request_states[r] = states
                            request_tokens[r] = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                            positions[r] += 1

            # warmup
            run_round_robin(2)
            _sync(device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _sync(device)

            with _PowerSampler(device) as sampler:
                started = time.perf_counter()
                run_round_robin(args.decode_tokens_per_request)
                _sync(device)
                elapsed = time.perf_counter() - started

            total_tokens = n_requests * args.decode_tokens_per_request
            entry = {
                "aggregate_tokens_per_second": total_tokens / elapsed,
                "per_request_tokens_per_second": args.decode_tokens_per_request / elapsed,
                "elapsed_seconds": elapsed,
                "mean_watts": sampler.mean_watts(),
                "peak_memory_bytes": peak_memory_bytes(device),
            }
            results["by_resident_request_count"][str(n_requests)] = entry
            mem_str = f"{entry['peak_memory_bytes']/1e9:.2f}GB" if entry["peak_memory_bytes"] is not None else "n/a"
            print(f"[N={n_requests}] {entry['aggregate_tokens_per_second']:.1f} aggregate tok/s "
                  f"peak_mem={mem_str}", flush=True)

            del request_states, request_tokens
        except torch.cuda.OutOfMemoryError as exc:
            results["by_resident_request_count"][str(n_requests)] = {"status": "OOM", "detail": str(exc)[:300]}
            print(f"[N={n_requests}] OOM: {exc}", flush=True)
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
