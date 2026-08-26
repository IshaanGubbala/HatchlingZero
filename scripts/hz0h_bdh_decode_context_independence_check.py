#!/usr/bin/env python3
"""Diagnostic, not a promoted-architecture benchmark: is BDH's real O(1)
streaming decode throughput actually flat across context (as Tier 0 item
3's original benchmark found, ~69.5 tok/s at every context 128-65536),
or does it degrade with context (as the newer multi-model decode
benchmarks tonight -- hz0h_bdh_subspace_decoder_decode_benchmark.py,
hz0h_bdh_vb_subspace_decoder_decode_benchmark.py -- both showed, BDH
alone dropping 68.9->59.9->40.0->39.9 tok/s)? Per-step decode compute is
provably O(1) in context (state shape is fixed regardless of how many
tokens contributed to it; RoPE position value doesn't change any tensor
shape) -- so a real degradation would mean something is wrong with the
architecture's own O(1) claim, worth taking seriously, not dismissing.

Two real candidate explanations tested here: (1) GPU allocator/thermal
carryover from holding 2-3 full models resident + large untimed prefill
calls processed back-to-back with no cache-clearing between context
iterations (a benchmark-script artifact, not an architecture property),
(2) a genuine property of the real streaming implementation. Isolates by
running ONE model only, with explicit torch.cuda.empty_cache() +
reset_peak_memory_stats() + a short cooldown between context points, and
by running the SAME context TWICE (once early, once late in the sweep,
interleaved) to see if it's context-dependent or order/carryover-dependent.
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
from scripts.hz0h_inference_benchmark import _PowerSampler, _sync, peak_memory_bytes, reset_peak_memory


def measure(model, prompt, max_new_tokens, device, prefill_chunk_length):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[128, 2048, 16384, 65536])
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--prefill-chunk-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cooldown-seconds", type=float, default=3.0)
    args = parser.parse_args()

    from scripts.hz0h_bdh_width_flop_frontier_local import pick_device
    device = pick_device(args.device)

    torch.manual_seed(args.seed)
    config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0)
    model = BDH(config).to(device=device, dtype=torch.bfloat16)
    model.attn.freqs = model.attn.freqs.to(torch.float32)
    model.eval()

    # Test order: ascending, then descending (repeat) -- if numbers depend
    # only on context (not on how much prior work happened), the ascending
    # and descending passes for the SAME context should roughly agree. If
    # they disagree (e.g. context=128 is fast ascending but slow on the
    # way back down after 65536), that's carryover/thermal, not context.
    order = list(args.context_lengths) + list(reversed(args.context_lengths))

    results = {"device": str(device), "dtype": "bfloat16", "single_model_isolated": True,
               "config": vars(args) | {"out": str(args.out)}, "runs": []}

    for i, context_length in enumerate(order):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        reset_peak_memory(device)
        time.sleep(args.cooldown_seconds)
        torch.manual_seed(args.seed)
        prompt = torch.randint(0, 256, (1, context_length), device=device)
        r = measure(model, prompt, args.decode_tokens, device, args.prefill_chunk_length)
        r["peak_memory_bytes"] = peak_memory_bytes(device)
        r["context_length"] = context_length
        r["pass_order_index"] = i
        results["runs"].append(r)
        mem_str = f"{r['peak_memory_bytes']/1e9:.2f}GB" if r["peak_memory_bytes"] is not None else "n/a"
        watts_str = f"{r['mean_watts']:.0f}W" if r["mean_watts"] is not None else "n/a"
        print(f"[pass {i}] context={context_length} {r['tokens_per_second']:.1f} tok/s "
              f"{watts_str} peak_mem={mem_str}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
