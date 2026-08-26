#!/usr/bin/env python3
"""Real multi-GPU test of Phase E's N-axis tensor-parallel decode
(reference/hz0h_bdh_vb_subspace_decoder_tensor_parallel_torch.py),
whose sharding decomposition was already verified bit-exact via a
single-device simulation (all-reduce = torch.sum over a Python list of
shards). This script replaces that simulation with REAL
torch.distributed.all_reduce calls over NCCL across `tp` real, separate
GPUs, launched via `torchrun --nproc_per_node=<tp>`.

Two things get measured for real here, not simulated: (1) correctness
-- does the real NCCL-based sharded decode step still match the real
unsharded single-GPU decode step, now that floating-point summation
order and real network communication are involved, not just a Python
sum(); (2) real wall-clock decode throughput at tp=1 vs tp=2 (and
tp=4 if enough GPUs are available), against the real single-GPU
baseline, to see whether the promised near-linear speedup from
sharding N (small persistent all-reduce, everything else fully local)
actually materializes on real hardware -- not assumed from the
simulated-correctness result alone.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_vb_subspace_decoder_stream_torch import bdh_vb_subspace_decoder_stream_chunk, bdh_vb_subspace_decoder_stream_prefill_chunked
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from reference.hz0h_bdh_vb_torch import init_bdh_vb_states


def build_shard(model: BDHVBSubspaceDecoder, rank: int, tp: int) -> dict:
    c = model.config
    nh, D = c.n_head, c.n_embd
    N = D * c.mlp_internal_dim_multiplier // nh
    assert N % tp == 0
    n_per_shard = N // tp
    assert n_per_shard % 2 == 0, "RoPE pairs adjacent coordinates -- shard boundaries must be even"
    lo, hi = rank * n_per_shard, (rank + 1) * n_per_shard
    return {
        "encoder": model.encoder[:, :, lo:hi].contiguous(),
        "encoder_v": model.encoder_v[:, :, lo:hi].contiguous(),
        "decoder_up": model.decoder_up.view(nh, N, -1)[:, lo:hi, :].reshape(-1, model.decoder_up.shape[-1]).contiguous(),
        "P": model.P, "O": model.O, "decoder_down": model.decoder_down,
        "freqs": model.attn.freqs[..., lo:hi].contiguous(),
        "n_per_shard": n_per_shard,
    }


def dist_sharded_decode_step(model: BDHVBSubspaceDecoder, shard: dict, shard_states: list[torch.Tensor],
                              idx_chunk: torch.Tensor, start_position: int, group) -> tuple[list[torch.Tensor], torch.Tensor]:
    c = model.config
    B, L = idx_chunk.shape
    D = c.n_embd
    x = model.embed(idx_chunk).unsqueeze(1)
    x = model.ln(x)

    new_states = []
    for level in range(c.n_layer):
        v_bottleneck = x @ model.P

        x_latent_i = x @ shard["encoder"]
        x_sparse_i = F.relu(x_latent_i)
        positions = torch.arange(start_position, start_position + L, device=x.device, dtype=shard["freqs"].dtype).view(1, 1, L, 1)
        r_phases_i = positions * shard["freqs"]
        QR_i = model.attn.rope(r_phases_i, x_sparse_i)
        KR_i = QR_i
        cross = QR_i @ shard_states[level]
        dist.all_reduce(cross, op=dist.ReduceOp.SUM, group=group)  # real (1): small, (B,nh,T,d_state)

        yKV_bottleneck = cross
        yKV = model.ln(yKV_bottleneck @ model.O)

        y_latent_i = yKV @ shard["encoder_v"]
        y_sparse_i = F.relu(y_latent_i)
        xy_sparse_i = model.drop(x_sparse_i * y_sparse_i)
        alpha = torch.matmul(xy_sparse_i, shard["decoder_up"].view(c.n_head, shard["n_per_shard"], -1)).sum(dim=1, keepdim=True)
        dist.all_reduce(alpha, op=dist.ReduceOp.SUM, group=group)  # real (2): tiny, (B,1,T,r)

        chunk_contribution_i = KR_i.mT @ v_bottleneck  # purely local, no communication
        new_states.append(shard_states[level] + chunk_contribution_i)

        yMLP = alpha @ model.decoder_down
        y = model.ln(yMLP)
        x = model.ln(x + y)

    logits = x.view(B, L, D) @ model.lm_head
    return new_states, logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--prefill-chunk-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    out_path = Path(args.out)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(args.seed)
    config = BDHVBSubspaceDecoderConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                                         mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
                                         d_state=args.d_state, subspace_rank=args.subspace_rank)
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.bfloat16).eval()
    shard = build_shard(model, rank, world_size)

    torch.manual_seed(args.seed)
    prompt = torch.randint(0, 256, (1, args.context_length), device=device)

    # --- correctness: sharded (this rank's contribution) vs the real unsharded single-GPU decode step ---
    with torch.no_grad():
        # every rank independently builds the SAME full unsharded model + runs the SAME real prefill/decode,
        # then compares against its own sharded computation -- real, not simulated, communication included.
        full_states = init_bdh_vb_states(model, 1, device=device)
        full_states, full_logits = bdh_vb_subspace_decoder_stream_prefill_chunked(model, prompt, chunk_length=args.prefill_chunk_length, states=full_states)
        full_token = torch.argmax(full_logits[:, -1, :], dim=-1, keepdim=True)

        n_per_shard = shard["n_per_shard"]
        lo = rank * n_per_shard
        # slice the PRE-step state (before it gets advanced below) -- a real bug caught here on
        # the first real-GPU run: reusing the post-step `full_states` for slicing fed the sharded
        # computation the wrong (already-advanced) state, not a flaw in the sharding math itself.
        shard_states = [full_states[level][:, :, lo:lo + n_per_shard, :].contiguous() for level in range(args.n_layer)]

        _, full_step_logits = bdh_vb_subspace_decoder_stream_chunk(model, full_states, full_token, start_position=args.context_length)
        group = dist.new_group(list(range(world_size)))
        _, sharded_step_logits = dist_sharded_decode_step(model, shard, shard_states, full_token, args.context_length, group)

        diff = float((full_step_logits - sharded_step_logits).abs().max())

    if rank == 0:
        print(f"[rank 0] correctness: real NCCL sharded decode step vs real unsharded decode step, max diff = {diff}", flush=True)

    # --- real throughput at this world_size ---
    with torch.no_grad():
        states = init_bdh_vb_states(model, 1, device=device)
        states, logits = bdh_vb_subspace_decoder_stream_prefill_chunked(model, prompt, chunk_length=args.prefill_chunk_length, states=states)
        token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        n_per_shard = shard["n_per_shard"]
        lo = rank * n_per_shard
        shard_states = [states[level][:, :, lo:lo + n_per_shard, :].contiguous() for level in range(args.n_layer)]

        def decode(n_tokens):
            nonlocal shard_states, token
            position = args.context_length
            for _ in range(n_tokens):
                shard_states, logits = dist_sharded_decode_step(model, shard, shard_states, token, position, group)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                position += 1

        dist.barrier()
        decode(4)  # warmup
        torch.cuda.synchronize()
        dist.barrier()

        started = time.perf_counter()
        decode(args.decode_tokens)
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - started

    if rank == 0:
        tok_per_sec = args.decode_tokens / elapsed
        print(f"[rank 0] world_size={world_size} real NCCL decode: {tok_per_sec:.2f} tok/s ({elapsed:.4f}s for {args.decode_tokens} tokens)", flush=True)
        result = {"world_size": world_size, "tokens_per_second": tok_per_sec, "elapsed_seconds": elapsed,
                  "correctness_max_diff": diff, "config": vars(args)}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[done] wrote {out_path}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
