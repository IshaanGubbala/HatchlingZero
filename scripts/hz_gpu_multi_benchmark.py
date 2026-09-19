"""Real multi-GPU throughput test via DistributedDataParallel, same
model/production shape as hz_gpu_batch_sweep.py (286.5M params,
chat_only, vocab=8192). Launch with torchrun:

    torchrun --nproc_per_node=2 scripts/hz_gpu_multi_benchmark.py \
        --per-gpu-batch-size 192 --k 32 --precision bf16 --compile

Each rank runs its own local batch (data-parallel), gradients
all-reduce via DDP after backward -- real aggregate tokens/sec is
per-gpu tok/s * world_size MINUS whatever all-reduce communication
overhead actually costs (the real question this script answers: does
that overhead eat the theoretical linear scaling or not)."""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=3072)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--block-mixer-heads", type=int, default=8)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--per-gpu-batch-size", type=int, default=192)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--timed-steps", type=int, default=30)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    torch.manual_seed(7)  # same seed on every rank -> identical init, DDP requires this
    model = HZLanguageModel(vocab_size=args.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                            workspace_slots=args.workspace_slots, n_rounds_l1=2,
                            block_mixer_heads=args.block_mixer_heads, chat_only=True).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    if args.compile:
        model.module.lm_forward_blocked = torch.compile(model.module.lm_forward_blocked, mode=args.compile_mode)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # different data per rank (real data-parallel split), same shape everywhere
    torch.manual_seed(7 + local_rank)
    ids = torch.randint(0, args.vocab_size, (args.per_gpu_batch_size, args.seq_len), device=local_rank)
    target = ids[:, 1:]
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    def step():
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(args.precision == "bf16")):
            logits = model.module.lm_forward_blocked(ids, block_size=args.k)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()  # DDP all-reduces gradients here automatically
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(args.warmup_steps):
        step()
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    t0 = time.time()
    for _ in range(args.timed_steps):
        step()
    torch.cuda.synchronize(local_rank)
    dist.barrier()
    elapsed = time.time() - t0

    local_examples_per_sec = (args.per_gpu_batch_size * args.timed_steps) / elapsed
    local_tokens_per_sec = local_examples_per_sec * args.seq_len

    # aggregate across ranks (real global throughput, not just rank-0's local number)
    tensor = torch.tensor([local_tokens_per_sec], device=local_rank)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_tokens_per_sec = tensor.item()

    if local_rank == 0:
        peak_mem_gb = torch.cuda.max_memory_allocated(local_rank) / (1024 ** 3)
        print(f"world_size={world_size} per_gpu_batch={args.per_gpu_batch_size} "
             f"local_tok/s={local_tokens_per_sec:,.0f} global_tok/s={global_tokens_per_sec:,.0f} "
             f"peak_mem_gb={peak_mem_gb:.2f} elapsed={elapsed:.2f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
