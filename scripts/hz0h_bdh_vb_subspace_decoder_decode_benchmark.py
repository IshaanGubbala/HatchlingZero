#!/usr/bin/env python3
"""Real decode-throughput benchmark for the compound VB frozen-identity
(d_state=624) + subspace-decoder-warmstart (r=64) architecture
(results/local/hz0h_vb_subspace_decoder_d624_r64.json, val_loss=1.7907,
beats both individual components on quality). Same real O(1)-state
streaming methodology as scripts/hz0h_bdh_subspace_decoder_decode_benchmark.py
(bdh_stream_chunk vs bdh_vb_subspace_decoder_stream_chunk), extended to
also compare against VB-alone streaming decode
(reference/hz0h_bdh_vb_torch.py's bdh_vb_stream_chunk) so this run
answers: does the compound model's decode speed reflect BOTH real
savings (decoder weight 36.7x smaller AND per-layer state 4x smaller at
d_state=624/D=2496), or does one saving dominate/mask the other.
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
from reference.hz0h_bdh_vb_torch import BDHVB, BDHVBConfig, bdh_vb_stream_chunk, bdh_vb_stream_prefill_chunked, init_bdh_vb_states
from scripts.hz0h_inference_benchmark import _PowerSampler, _sync, peak_memory_bytes


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
            torch.cuda.empty_cache()  # real fix, 2026-08-25: see scripts/hz0h_bdh_decode_context_independence_check.py -- prefill's fragmented allocator state was bleeding into the timed decode region, making decode falsely look context-dependent
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
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[128, 2048, 16384, 65536])
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
    vb_config = BDHVBConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                             mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
                             d_state=args.d_state)
    vb_model = BDHVB(vb_config).to(device=device, dtype=torch.bfloat16).eval()

    torch.manual_seed(args.seed)
    compound_config = BDHVBSubspaceDecoderConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                                                  mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
                                                  d_state=args.d_state, subspace_rank=args.subspace_rank)
    compound_model = BDHVBSubspaceDecoder(compound_config).to(device=device, dtype=torch.bfloat16).eval()

    state_elems_dense = args.n_head * (args.n_embd * args.mult // args.n_head) * args.n_embd
    state_elems_vb = args.n_head * (args.n_embd * args.mult // args.n_head) * args.d_state
    decoder_bytes_dense = bdh_model.decoder.numel() * bdh_model.decoder.element_size()
    decoder_bytes_subspace = (compound_model.decoder_up.numel() * compound_model.decoder_up.element_size()
                               + compound_model.decoder_down.numel() * compound_model.decoder_down.element_size())

    results = {
        "device": str(device), "dtype": "bfloat16",
        "note": "untrained execution-speed diagnostic -- decode tok/s and memory only, not a quality claim (quality already established: results/local/hz0h_vb_subspace_decoder_d624_r64.json, val_loss=1.7907, beats both individual components). Real O(1)-state streaming decode path for all three models.",
        "config": {"n_embd": args.n_embd, "mult": args.mult, "n_layer": args.n_layer, "n_head": args.n_head,
                    "d_state": args.d_state, "subspace_rank": args.subspace_rank, "decode_tokens": args.decode_tokens, "seed": args.seed},
        "per_layer_state_elems": {"dense_bdh": state_elems_dense, "vb_and_compound": state_elems_vb,
                                   "reduction_factor": state_elems_dense / state_elems_vb},
        "decoder_weight_bytes": {"dense": decoder_bytes_dense, "subspace_and_compound": decoder_bytes_subspace,
                                  "reduction_factor": decoder_bytes_dense / decoder_bytes_subspace},
        "by_context_length": {},
    }

    for context_length in args.context_lengths:
        torch.manual_seed(args.seed)
        prompt = torch.randint(0, 256, (1, context_length), device=device)
        try:
            bdh_decode = _measure(bdh_stream_prefill_chunked, bdh_stream_chunk, init_bdh_states,
                                   bdh_model, prompt, args.decode_tokens, device, args.prefill_chunk_length)
            bdh_decode["peak_memory_bytes"] = peak_memory_bytes(device)

            vb_decode = _measure(bdh_vb_stream_prefill_chunked, bdh_vb_stream_chunk, init_bdh_vb_states,
                                  vb_model, prompt, args.decode_tokens, device, args.prefill_chunk_length)
            vb_decode["peak_memory_bytes"] = peak_memory_bytes(device)

            compound_decode = _measure(bdh_vb_subspace_decoder_stream_prefill_chunked, bdh_vb_subspace_decoder_stream_chunk, init_bdh_vb_states,
                                        compound_model, prompt, args.decode_tokens, device, args.prefill_chunk_length)
            compound_decode["peak_memory_bytes"] = peak_memory_bytes(device)

            results["by_context_length"][str(context_length)] = {
                "bdh_decode_streaming": bdh_decode,
                "vb_decode_streaming": vb_decode,
                "compound_decode_streaming": compound_decode,
                "vb_over_bdh_speedup": vb_decode["tokens_per_second"] / bdh_decode["tokens_per_second"],
                "compound_over_bdh_speedup": compound_decode["tokens_per_second"] / bdh_decode["tokens_per_second"],
                "compound_over_vb_speedup": compound_decode["tokens_per_second"] / vb_decode["tokens_per_second"],
            }
            r = results["by_context_length"][str(context_length)]
            print(f"[context={context_length}] BDH {bdh_decode['tokens_per_second']:.1f} tok/s | "
                  f"VB {vb_decode['tokens_per_second']:.1f} tok/s ({r['vb_over_bdh_speedup']:.3f}x) | "
                  f"Compound {compound_decode['tokens_per_second']:.1f} tok/s ({r['compound_over_bdh_speedup']:.3f}x vs BDH, {r['compound_over_vb_speedup']:.3f}x vs VB)", flush=True)
        except torch.cuda.OutOfMemoryError as exc:
            results["by_context_length"][str(context_length)] = {"status": "OOM", "detail": str(exc)}
            print(f"[context={context_length}] OOM: {exc}", flush=True)
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
