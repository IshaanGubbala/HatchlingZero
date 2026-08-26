#!/usr/bin/env python3
"""Phase B step 1: real per-operator profiling of the compound model's
decode step at B=1/2/4, using torch.profiler, to find out WHICH
operation actually causes the 156->75 aggregate-tok/s collapse when
physical batch size grows -- rather than guessing at a rewrite. Real
candidates named in the investigation: the encoder/encoder_v matmuls
(x @ model.encoder, shape (nh,D,N), broadcast over (B,1) leading dims --
implicit batched-GEMM dispatch, nh separate thin-M calls rather than one
packed GEMM) and the state mutation
(`new_states.append(prefix_state + chunk_contribution)`, which allocates
a full new state-sized tensor every decode step instead of updating a
preallocated buffer in place).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

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
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--with-stack", action="store_true")
    args = parser.parse_args()

    from scripts.hz0h_bdh_width_flop_frontier_local import pick_device
    device = pick_device(args.device)
    assert device.type == "cuda", "real op-level CUDA time profiling needs a real CUDA device"

    torch.manual_seed(args.seed)
    config = BDHVBSubspaceDecoderConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                                         mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
                                         d_state=args.d_state, subspace_rank=args.subspace_rank)
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.bfloat16).eval()

    results = {"device": str(device), "dtype": "bfloat16",
               "note": "real torch.profiler op-level CUDA time breakdown of the compound decode step, per batch size -- finds WHICH op's cost scales worst with batch, before rewriting anything.",
               "config": vars(args) | {"out": str(args.out)},
               "by_batch_size": {}}

    for batch_size in args.batch_sizes:
        torch.manual_seed(args.seed)
        prompt = torch.randint(0, 256, (batch_size, args.context_length), device=device)

        with torch.no_grad():
            states = init_bdh_vb_states(model, batch_size, device=device)
            states, logits = bdh_vb_subspace_decoder_stream_prefill_chunked(
                model, prompt, chunk_length=args.context_length, states=states)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        # warmup
        with torch.no_grad():
            position = args.context_length
            for _ in range(3):
                states, logits = bdh_vb_subspace_decoder_stream_chunk(model, states, token, start_position=position)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                position += 1
        torch.cuda.synchronize()

        with torch.no_grad(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=args.with_stack) as prof:
            for _ in range(args.profile_steps):
                states, logits = bdh_vb_subspace_decoder_stream_chunk(model, states, token, start_position=position)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                position += 1
            torch.cuda.synchronize()

        if args.with_stack:
            for evt in prof.key_averages(group_by_stack_n=5):
                if evt.key in ("aten::reshape", "aten::clone", "aten::copy_") and evt.device_time_total > 0:
                    print(f"  --- {evt.key} ({evt.device_time_total/1000:.2f}ms total) call stack ---", flush=True)
                    for line in (evt.stack or [])[:5]:
                        print(f"      {line}", flush=True)

        key_avgs = prof.key_averages()
        sorted_by_cuda = sorted(key_avgs, key=lambda e: e.device_time_total, reverse=True)[:args.top_n]
        op_table = [
            {"name": e.key, "cuda_time_total_us": e.device_time_total, "cuda_time_avg_us": e.device_time_total / max(e.count, 1),
             "cpu_time_total_us": e.cpu_time_total, "count": e.count}
            for e in sorted_by_cuda
        ]
        total_cuda_time = sum(e.device_time_total for e in key_avgs)
        results["by_batch_size"][str(batch_size)] = {"total_cuda_time_us": total_cuda_time, "top_ops": op_table}
        print(f"[batch={batch_size}] total_cuda_time={total_cuda_time/1000:.2f}ms over {args.profile_steps} steps "
              f"({total_cuda_time/args.profile_steps/1000:.3f}ms/step)", flush=True)
        for op in op_table[:8]:
            print(f"    {op['name'][:60]:60s} {op['cuda_time_total_us']/1000:8.3f}ms total  {op['count']:4d} calls  "
                  f"{op['cuda_time_avg_us']:8.1f}us/call", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
