"""HZ2 fixed-patch pilot runner, per plans/HZ_Mixed_Depth_Pilot_2026-09-15.md
sections 3-4.

Scope of THIS pass: the systems-cost measurement (section 4's "before
quality runs: 20 warmup and 100 timed full optimizer steps... if patching
improves neither speed nor memory, diagnose before paying for training").
Uses SYNTHETIC random bytes -- a speed/memory benchmark doesn't need real
corpus data, only quality training does (a separate, larger, not-yet-built
piece: real data manifest loading, LR sweep, 64M/512M-byte screening runs).

Runs on CPU/MPS locally with small shapes for a correctness/plausibility
check of the runner itself (no GPU spend) -- the plan's own real production
shapes (D=768/2496-style widths) need actual GPU to be meaningful for
timing, and running those is a real dispatch decision requiring explicit
approval (see CLAUDE.md's Runpod dispatch pattern), not something this
script does on its own.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.hz2_patch_model_torch import HZ2Config, HZ2PatchModel

# Plan section 2's exact production shape.
PRODUCTION_CONFIG_KWARGS = dict(
    vocab_size=256, byte_width=192, byte_ffn=512, byte_heads=3, byte_head_dim=64,
    core_width=768, core_ffn=2048, core_q_heads=12, core_kv_heads=3, core_head_dim=64,
    core_bank_repeats=4, core_attn_window_patches=128, decoder_attn_window_bytes=256,
    decoder_q_heads=3, decoder_kv_heads=1,
)


def build_config(arm_k: int, production: bool, **overrides) -> HZ2Config:
    kwargs = dict(PRODUCTION_CONFIG_KWARGS) if production else dict(
        vocab_size=256, byte_width=32, byte_ffn=64, byte_heads=2, byte_head_dim=16,
        core_width=48, core_ffn=96, core_q_heads=4, core_kv_heads=2, core_head_dim=12,
        core_bank_repeats=2, core_attn_window_patches=8, decoder_attn_window_bytes=16,
        decoder_q_heads=2, decoder_kv_heads=1,
    )
    kwargs.update(overrides)
    kwargs["K"] = arm_k
    return HZ2Config(**kwargs)


def synchronize(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def peak_memory_bytes(device: str) -> dict:
    if device == "cuda":
        return {"peak_allocated": torch.cuda.max_memory_allocated(),
               "peak_reserved": torch.cuda.max_memory_reserved()}
    if device == "mps":
        # torch.mps exposes driver_allocated_memory, not a CUDA-style peak
        # counter -- report what's actually available rather than fake a
        # "peak" MPS doesn't track the same way.
        return {"current_allocated": torch.mps.driver_allocated_memory()}
    return {}


def benchmark_arm(config: HZ2Config, device: str, batch_size: int, seq_len: int,
                  warmup_steps: int, timed_steps: int, lr: float = 1e-3, seed: int = 7):
    torch.manual_seed(seed)
    model = HZ2PatchModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=0.1)
    n_params = model.param_count()

    def one_step():
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        optimizer.zero_grad()
        _, loss = model(idx, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    for _ in range(warmup_steps):
        one_step()
    synchronize(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    last_loss = None
    for _ in range(timed_steps):
        last_loss = one_step()
    synchronize(device)
    elapsed = time.perf_counter() - start

    bytes_per_step = batch_size * seq_len
    return {
        "K": config.K, "device": device, "batch_size": batch_size, "seq_len": seq_len,
        "warmup_steps": warmup_steps, "timed_steps": timed_steps,
        "elapsed_seconds": elapsed, "ms_per_step": 1000 * elapsed / timed_steps,
        "bytes_per_second": bytes_per_step * timed_steps / elapsed,
        "trainable_parameters": n_params, "last_loss": last_loss,
        "memory": peak_memory_bytes(device),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=int, choices=[1, 4, 8], required=True, help="Patch size K.")
    parser.add_argument("--production-shape", action="store_true",
                        help="Use the plan's real D=768/byte_width=192 production config, not the tiny "
                             "local-debug shape. Only meaningful timing on a real GPU; on CPU/MPS this "
                             "checks the runner works at the real shape, not real throughput numbers.")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--timed-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = build_config(args.arm, args.production_shape)
    print(f"arm K={args.arm}  production_shape={args.production_shape}  device={args.device}  "
          f"batch={args.batch_size} seq_len={args.seq_len}")
    result = benchmark_arm(config, args.device, args.batch_size, args.seq_len,
                           args.warmup_steps, args.timed_steps, args.lr, args.seed)
    print(json.dumps(result, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
