#!/usr/bin/env python3
"""Real DDP (DistributedDataParallel) multi-GPU TRAINING throughput test
for the compound model, launched via `torchrun --nproc_per_node=<n>`.
Tonight's tensor-parallel work (Phase E) was decode-only -- this is the
separate question raised directly: does real multi-GPU (data-parallel,
not the N-axis-sharded decode scheme) actually speed up TRAINING, and
is it worth the extra hourly cost on faster cards (e.g. 2x RTX 5090)?

Real, bounded throughput comparison, not a full training run: each rank
trains the SAME real compound model (checkpointed forward, same
methodology as tonight's quality-check scripts) for a fixed number of
steps, DDP-wrapped so gradients real-sync via NCCL all-reduce every
step. Reports real global tokens/sec (world_size * per-GPU tok/s,
accounting for real DDP overhead, not assumed linear scaling) against a
single-GPU (world_size=1) baseline run with the SAME per-GPU batch size,
so the comparison isolates "does adding GPUs help" from "is this GPU
faster than a 4090" (that part is already measured separately by
tonight's decode benchmarks: 5090 tp=1 was 252.85 vs 4090's 159.98
tok/s, ~1.58x).

Real, known friction point tested here, not assumed to just work:
gradient checkpointing (torch.utils.checkpoint) combined with DDP is a
documented source of real issues (DDP's bucketed-gradient hooks can
conflict with checkpoint recomputation) -- this script surfaces any
real error rather than silently falling back to something else.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_vb_subspace_decoder_checkpointed_torch import bdh_vb_subspace_decoder_forward_checkpointed
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_combined_best_comparison import autocast_context, curriculum_stages, make_optimizer
from scripts.hz0h_factorized_curriculum_full_comparison import depth_at, lr_at, read_batch


class BDHVBSubspaceDecoderCheckpointedForward(BDHVBSubspaceDecoder):
    """Real fix for DDP compatibility: DistributedDataParallel registers its
    gradient-sync autograd hooks during the actual `model(...)` call
    (its own wrapped `.forward()`), not on a standalone function called
    against `model.module` afterward -- calling the checkpointed forward
    function directly on `.module` would silently skip DDP's gradient
    all-reduce. This subclass makes `.forward()` itself perform the real
    checkpointed round-loop, so `model(idx, targets=target)` (going
    through DDP's own `__call__`) triggers gradient sync correctly."""

    def forward(self, idx, targets=None):
        _, loss = bdh_vb_subspace_decoder_forward_checkpointed(self, idx, self.config.n_layer, targets)
        return None, loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8, help="per-GPU batch size")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    args = parser.parse_args()
    args.dtype = "bfloat16"
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
    assert args.depth == args.n_layer, "this benchmark's forward always runs full depth (no curriculum) -- pass --depth equal to --n-layer"
    model = BDHVBSubspaceDecoderCheckpointedForward(config).to(device=device, dtype=torch.float32)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = make_optimizer(model.parameters(), args, device)

    epochs = [0]
    # Each rank reads a DIFFERENT slice of the file (real data parallelism,
    # not every rank re-reading identical rows) -- skip rank*measure_window
    # lines up front so ranks don't train on the same tokens.
    with args.data.open() as handle:
        skip_lines = rank * (args.warmup_steps + args.measure_steps) * args.batch_size
        for _ in range(skip_lines):
            handle.readline()

        def step():
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args, device):
                _, loss = model(idx, targets=target)  # real DDP call -- gradient-sync hooks attach here
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            return float(loss)

        for _ in range(args.warmup_steps):
            step()
        torch.cuda.synchronize()
        dist.barrier()

        started = time.perf_counter()
        last_loss = None
        for _ in range(args.measure_steps):
            last_loss = step()
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - started

    tokens_per_gpu = args.measure_steps * args.batch_size * args.sequence_length
    per_gpu_tok_s = tokens_per_gpu / elapsed
    global_tok_s = per_gpu_tok_s * world_size

    if rank == 0:
        print(f"[rank 0] world_size={world_size} per_gpu_batch={args.batch_size} depth={args.depth} "
              f"per_gpu_tok/s={per_gpu_tok_s:.1f} GLOBAL_tok/s={global_tok_s:.1f} "
              f"elapsed={elapsed:.2f}s last_loss={last_loss:.4f}", flush=True)
        result = {"world_size": world_size, "per_gpu_batch_size": args.batch_size, "depth": args.depth,
                  "per_gpu_tokens_per_second": per_gpu_tok_s, "global_tokens_per_second": global_tok_s,
                  "elapsed_seconds": elapsed, "measure_steps": args.measure_steps, "last_loss": last_loss,
                  "config": vars(args) | {"out": str(args.out), "data": str(args.data)}}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[done] wrote {out_path}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
