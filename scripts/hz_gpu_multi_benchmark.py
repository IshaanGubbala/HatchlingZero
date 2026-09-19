"""Real multi-GPU throughput test via DistributedDataParallel, same
model/production shape as hz_gpu_batch_sweep.py (286.5M params,
chat_only, vocab=8192). Launch with torchrun:

    torchrun --nproc_per_node=2 scripts/hz_gpu_multi_benchmark.py \
        --per-gpu-batch-size 192 --k 32 --precision bf16 --compile

Each rank runs its own local batch (data-parallel), gradients
all-reduce via DDP after backward -- real aggregate tokens/sec is
per-gpu tok/s * world_size MINUS whatever all-reduce communication
overhead actually costs (the real question this script answers: does
that overhead eat the theoretical linear scaling or not).

2026-09-19: real fix + scaling-efficiency levers. The first version of
this script called `model.module.lm_forward_blocked(...)` directly
instead of going through DDP's own `forward()` -- that skips DDP's
`prepare_for_backward()` bucket-readiness bookkeeping. It happened to
produce sane-looking numbers (autograd hooks are parameter-attached,
not forward-path-attached, so gradient sync still basically worked),
but it's not the documented, correct usage and would silently break
`static_graph=True`. Fixed via BlockedForwardWrapper below, whose
`forward()` DDP actually wraps. Also adds three real tuning levers for
scaling efficiency: `--bucket-cap-mb` (fewer, larger all-reduce calls
instead of DDP's default 25MB many-small-buckets), `--static-graph`
(unlocks extra DDP overlap since this model's compute graph never
changes between steps), and `--grad-accum-steps` (syncs once every N
local steps via `no_sync()`, diluting the FIXED per-sync communication
cost over more compute -- the lever that should matter most at real
training lengths, not just this microbenchmark's 30 timed steps)."""
from __future__ import annotations

import argparse
import contextlib
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


class BlockedForwardWrapper(torch.nn.Module):
    """Real forward() DDP can hook into -- see module docstring for why
    calling lm_forward_blocked directly (bypassing this) was wrong."""

    def __init__(self, model: HZLanguageModel, block_size: int):
        super().__init__()
        self.model = model
        self.block_size = block_size

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.lm_forward_blocked(token_ids, block_size=self.block_size)


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
    parser.add_argument("--bucket-cap-mb", type=int, default=25,
                        help="DDP's default is 25MB (~46 buckets for this 286.5M-param model in fp32 "
                             "gradient space). A larger value means fewer, bigger all-reduce calls -- "
                             "real lever since NCCL has meaningful fixed latency per call.")
    parser.add_argument("--find-unused-parameters", action="store_true",
                        help="Required for this model unless --static-graph handles it instead -- "
                             "ws.read_x.k_proj never gets gradient (real, verified fast-path consequence).")
    parser.add_argument("--static-graph", action="store_true",
                        help="Tells DDP the compute graph never changes between iterations (true here -- "
                             "no dynamic control flow), unlocking extra communication/computation overlap "
                             "DDP can't safely do otherwise.")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Sync gradients once every N local steps (via DDP's no_sync() for the "
                             "N-1 non-syncing steps) instead of every step. Dilutes the FIXED per-sync "
                             "communication cost over more local compute -- should matter most at real "
                             "training lengths where many total steps happen, not just a short benchmark.")
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
    # Real, comprehensive dead-parameter list for lm_forward_blocked
    # training, found via TORCH_DISTRIBUTED_DEBUG=DETAIL (2026-09-19),
    # NOT assumed: ws.read_x.k_proj (softmax over one source has zero
    # derivative -- the known fast-path consequence), lm_rq/rk/rv (the
    # old unpacked per-token read projections, superseded by
    # _packed_lm_read, which lm_forward_blocked actually calls), and
    # every parameter in mem (S's persistent-memory submodule) --
    # lm_forward_blocked's own docstring already states S stays
    # untouched-at-init throughout, so mem.update() is never called and
    # none of its parameters ever enter the forward graph. Freezing all
    # of them removes the need for DDP's find_unused_parameters/
    # static_graph handling -- real, measured fix: find_unused_parameters
    # alone was cutting per-GPU throughput roughly in half via its real
    # per-iteration autograd-graph traversal, confirmed present even
    # without torch.compile, so this isn't a compile artifact.
    model.ws.read_x.k_proj.weight.requires_grad_(False)
    model.lm_rq.weight.requires_grad_(False)
    model.lm_rk.weight.requires_grad_(False)
    model.lm_rv.weight.requires_grad_(False)
    for p in model.mem.parameters():
        p.requires_grad_(False)
    if args.compile:
        model.lm_forward_blocked = torch.compile(model.lm_forward_blocked, mode=args.compile_mode)
    wrapper = BlockedForwardWrapper(model, args.k)
    # ws.read_x.k_proj never receives gradient (a real, verified,
    # disclosed consequence of the single-source cross-attention fast
    # path -- see tests/test_hz_block_recurrence.py). Without
    # find_unused_parameters=True, DDP's Reducer throws "Expected to have
    # finished reduction..." on the first real backward through the
    # correct forward() path (confirmed 2026-09-19: the crash only
    # appeared once this script stopped bypassing DDP's forward()).
    # static_graph=True's own docs claim it discovers unused parameters
    # via a one-time first-iteration profiling pass instead of
    # find_unused_parameters' per-step check -- real, untested-until-now
    # claim, so pass both flags through as given rather than assume
    # they conflict; let PyTorch itself error if they truly don't combine.
    ddp_model = DDP(wrapper, device_ids=[local_rank], bucket_cap_mb=args.bucket_cap_mb,
                    gradient_as_bucket_view=True, static_graph=args.static_graph,
                    find_unused_parameters=args.find_unused_parameters)
    opt = torch.optim.AdamW(ddp_model.parameters(), lr=3e-4)

    # different data per rank (real data-parallel split), same shape everywhere
    torch.manual_seed(7 + local_rank)
    ids = torch.randint(0, args.vocab_size, (args.per_gpu_batch_size, args.seq_len), device=local_rank)
    target = ids[:, 1:]
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    def step():
        opt.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum_steps):
            sync_ctx = ddp_model.no_sync() if micro < args.grad_accum_steps - 1 else contextlib.nullcontext()
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(args.precision == "bf16")):
                    logits = ddp_model(ids)
                    loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), target.reshape(-1))
                    loss = loss / args.grad_accum_steps
                loss.backward()  # DDP all-reduces on the LAST micro-step only, when sync_ctx is not no_sync()
        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
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

    # real examples processed = batch * grad_accum_steps * timed_steps (each "step" now does
    # grad_accum_steps local forward/backward passes before one optimizer update)
    local_examples_per_sec = (args.per_gpu_batch_size * args.grad_accum_steps * args.timed_steps) / elapsed
    local_tokens_per_sec = local_examples_per_sec * args.seq_len

    tensor = torch.tensor([local_tokens_per_sec], device=local_rank)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_tokens_per_sec = tensor.item()

    if local_rank == 0:
        peak_mem_gb = torch.cuda.max_memory_allocated(local_rank) / (1024 ** 3)
        print(f"world_size={world_size} per_gpu_batch={args.per_gpu_batch_size} "
             f"grad_accum={args.grad_accum_steps} bucket_cap_mb={args.bucket_cap_mb} "
             f"static_graph={args.static_graph} "
             f"local_tok/s={local_tokens_per_sec:,.0f} global_tok/s={global_tokens_per_sec:,.0f} "
             f"peak_mem_gb={peak_mem_gb:.2f} elapsed={elapsed:.2f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
