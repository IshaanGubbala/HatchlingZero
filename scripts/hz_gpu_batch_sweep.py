"""Real GPU batch-size/throughput sweep for the target HZ chat-model shape,
on whatever CUDA GPU this runs on (RunPod dispatch: 4090/A40/L40/etc, per
explicit request 2026-09-17 -- separate from this repo's Windows RTX3060
relay and separate from the "prefer RTX 5090" single-GPU dispatch default,
since the point here is comparing GPUs, not picking the fastest one).

Synthetic random token ids (no corpus dependency -- this measures pure
compute throughput, not data loading) at the real production shape:
d_model=3072, chat_only=True, vocab_size=8192 (the just-switched BPE
vocab), K=32 block recurrence. Sweeps batch size upward until CUDA OOM or
throughput plateaus, same diagnostic already run once on MPS locally
(hz_300m_committed_run2 investigation: ceiling ~42-44 examples/sec around
batch=16 on Apple GPU) -- this repeats that measurement on real CUDA
hardware for a real, comparable number.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402


def gpu_snapshot() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
        name, util, mem_used, mem_total, power = [x.strip() for x in out.split(",")]
        return {"name": name, "util_percent": float(util), "mem_used_mb": float(mem_used),
               "mem_total_mb": float(mem_total), "power_draw_w": float(power)}
    except Exception as e:
        return {"error": str(e)}


def run_one_batch_size(model, opt, batch_size, seq_len, vocab_size, block_size, device,
                       warmup_steps, timed_steps):
    torch.cuda.reset_peak_memory_stats(device)
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    target = ids[:, 1:]

    def step():
        logits = model.lm_forward_blocked(ids, block_size=block_size)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return loss.item()

    for _ in range(warmup_steps):
        step()
    torch.cuda.synchronize(device)

    mid_snapshot = None
    t0 = time.time()
    for i in range(timed_steps):
        step()
        if i == timed_steps // 2:
            torch.cuda.synchronize(device)
            mid_snapshot = gpu_snapshot()
    torch.cuda.synchronize(device)
    elapsed = time.time() - t0

    examples_per_sec = (batch_size * timed_steps) / elapsed
    tokens_per_sec = examples_per_sec * seq_len
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return {"batch_size": batch_size, "elapsed_seconds": elapsed, "examples_per_sec": examples_per_sec,
           "tokens_per_sec": tokens_per_sec, "peak_mem_gb": peak_mem_gb, "gpu_mid_run": mid_snapshot}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=3072)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--block-mixer-heads", type=int, default=8)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("results/local/hz_gpu_batch_sweep.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available on this machine -- this benchmark requires a real GPU.", flush=True)
        sys.exit(1)
    device = "cuda"
    print(f"GPU: {gpu_snapshot()}", flush=True)
    print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}", flush=True)

    torch.manual_seed(7)
    model = HZLanguageModel(vocab_size=args.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                            workspace_slots=args.workspace_slots, n_rounds_l1=2,
                            block_mixer_heads=args.block_mixer_heads, chat_only=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {total_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    results = []
    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        print(f"\n--- batch_size={bs} ---", flush=True)
        try:
            r = run_one_batch_size(model, opt, bs, args.seq_len, args.vocab_size, args.k, device,
                                   args.warmup_steps, args.timed_steps)
            print(f"  {r['examples_per_sec']:.1f} examples/sec, {r['tokens_per_sec']:,.0f} tokens/sec, "
                 f"peak_mem={r['peak_mem_gb']:.2f}GB, gpu_util_mid_run={r['gpu_mid_run']}", flush=True)
            results.append(r)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at batch_size={bs}: {e}", flush=True)
            torch.cuda.empty_cache()
            results.append({"batch_size": bs, "oom": True})
            break

    best = max((r for r in results if r.get("examples_per_sec")), key=lambda r: r["examples_per_sec"], default=None)
    print("\n=== summary ===")
    if best:
        print(f"best throughput: batch_size={best['batch_size']}  "
             f"{best['examples_per_sec']:.1f} examples/sec  {best['tokens_per_sec']:,.0f} tokens/sec  "
             f"peak_mem={best['peak_mem_gb']:.2f}GB")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"gpu": gpu_snapshot(), "torch_version": torch.__version__,
                                    "cuda_version": torch.version.cuda, "total_params": total_params,
                                    "config": vars(args) | {"out": str(args.out)},
                                    "results": results, "best": best}, indent=2, default=str) + "\n")
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
