#!/usr/bin/env python3
"""Isolates WHERE the compound model's real batch-scaling memory blowup
actually comes from. Real anomaly caught by a sharp user question: at
batch=4, the analytic persistent synaptic state is only
8 layers * 8 heads * 4992 * 624 * 2 bytes = 398.72MB/request
(~1.49GiB at B=4), but the batch-frontier benchmark
(results/local/hz0h_vb_subspace_decoder_batch_frontier.json) measured a
peak of 23.27GB at B=4 -- ~15.7x more than persistent state alone
explains. Two real candidate explanations, tested separately here:
(1) PREFILL's own transient activations -- the intra-chunk attention
term (QR @ KR.mT).tril(...) @ v_bottleneck materializes a real
(B, nh, chunk_length, chunk_length) score matrix per layer, which at
chunk_length=2048 is large and scales with B; x_sparse/y_sparse/xy_sparse
are each (B, 1, chunk_length, N) with N=4992, also real and batch-scaled.
None of this is persistent -- it's freed after prefill, in principle.
(2) the persistent per-layer state itself, which genuinely is B-scaled
and unavoidable given the architecture (this is the "fundamental"
half of the user's own framing).

Measures peak memory in THREE separate phases per batch size: (a) model
construction, (b) prefill alone (fresh peak-stats reset right before),
(c) decode-only steady-state (fresh peak-stats reset right after prefill,
before the timed decode steps) -- so each phase's contribution is
isolated, not conflated into one whole-run peak number the way the
batch-frontier benchmark measured it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_vb_subspace_decoder_stream_torch import bdh_vb_subspace_decoder_stream_chunk, bdh_vb_subspace_decoder_stream_prefill_chunked
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from reference.hz0h_bdh_vb_torch import init_bdh_vb_states


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
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--prefill-chunk-length", type=int, default=2048)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from scripts.hz0h_bdh_width_flop_frontier_local import pick_device
    device = pick_device(args.device)
    assert device.type == "cuda", "this diagnostic needs real CUDA peak-memory-stats semantics"

    torch.manual_seed(args.seed)
    config = BDHVBSubspaceDecoderConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                                         mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
                                         d_state=args.d_state, subspace_rank=args.subspace_rank)

    N = args.n_embd * args.mult // args.n_head
    state_bytes_per_request = args.n_layer * args.n_head * N * args.d_state * 2  # bf16

    results = {"device": str(device), "dtype": "bfloat16",
               "note": "isolates prefill-transient-activation memory from persistent decode-state memory, per batch size, for the compound model. Not a quality or throughput claim.",
               "config": vars(args) | {"out": str(args.out)},
               "analytic_state_bytes_per_request": state_bytes_per_request,
               "by_batch_size": {}}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.bfloat16).eval()
    model_load_peak = torch.cuda.max_memory_allocated()
    print(f"[model] peak_allocated={model_load_peak/1e9:.3f}GB (weights + construction overhead)", flush=True)

    for batch_size in args.batch_sizes:
        torch.manual_seed(args.seed)
        prompt = torch.randint(0, 256, (batch_size, args.context_length), device=device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            states = init_bdh_vb_states(model, batch_size, device=device)
            states, logits = bdh_vb_subspace_decoder_stream_prefill_chunked(
                model, prompt, chunk_length=args.prefill_chunk_length, states=states)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize()
        prefill_peak = torch.cuda.max_memory_allocated()
        prefill_reserved = torch.cuda.max_memory_reserved()

        # Free everything prefill-transient could have left resident except
        # `states`/`token` themselves, then measure ONLY the persistent
        # state's real resident footprint before any decode step runs.
        del logits, prompt
        torch.cuda.synchronize()
        state_only_allocated = torch.cuda.memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            position = args.context_length
            for _ in range(args.decode_tokens):
                states, logits = bdh_vb_subspace_decoder_stream_chunk(model, states, token, start_position=position)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                position += 1
        torch.cuda.synchronize()
        decode_peak = torch.cuda.max_memory_allocated()
        decode_reserved = torch.cuda.max_memory_reserved()

        entry = {
            "prefill_peak_allocated_bytes": prefill_peak,
            "prefill_peak_reserved_bytes": prefill_reserved,
            "state_only_allocated_bytes_after_prefill": state_only_allocated,
            "analytic_state_bytes_this_batch": state_bytes_per_request * batch_size,
            "decode_only_peak_allocated_bytes": decode_peak,
            "decode_only_peak_reserved_bytes": decode_reserved,
        }
        results["by_batch_size"][str(batch_size)] = entry
        print(f"[batch={batch_size}] prefill_peak={prefill_peak/1e9:.3f}GB "
              f"state_only={state_only_allocated/1e9:.3f}GB (analytic={entry['analytic_state_bytes_this_batch']/1e9:.3f}GB) "
              f"decode_peak={decode_peak/1e9:.3f}GB decode_reserved={decode_reserved/1e9:.3f}GB", flush=True)

        del states, token
        torch.cuda.empty_cache()

    results["model_construction_peak_allocated_bytes"] = model_load_peak
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
