#!/usr/bin/env python3
"""Retests Part 6's jump-operator prototype
(`docs/restart/hz0h_inherited_choices_audit_results.md`) at real
production scale, not the original 128-dim/1.5M-token toy prototype.

Real motivation: Part 6 found "settle 4 real iterations, then jump the
rest" gets 1.9x real wall-clock speedup for only +0.029 validation loss
-- the strongest efficiency signal in the whole audit -- but only ever
tested on a tiny model (`n_embd=128`, 1.5M tokens). This session's
capstone infrastructure (bf16 autocast, gradient checkpointing, int8
optimizer state, all proven working on real hardware for `raw_bdh` at
300M params) makes a real production-scale retest possible for the
first time.

Deliberately isolates the jump operator from `combined_best`'s
`softmax_scaled` attention (which diverged to NaN at this scale,
Windows dispatch 2026-08-20) -- this script trains a PLAIN-attention
teacher (same recipe as the capstone's already-proven `raw_bdh` arm),
so any result here is about the jump operator specifically, not
confounded with the separate, still-open softmax_scaled bug.

Reuses `make_optimizer`/`autocast_context`/`curriculum_stages` from
`scripts/hz0h_bdh_combined_best_comparison.py` rather than duplicating
them, so the bf16/checkpointing/int8-optimizer behavior can't drift
between the two scripts.

Method:
1. Train a real BDH teacher at production scale (plain attention,
   checkpointed, bf16, int8-optimizer-capable) -- same recipe as the
   capstone's `raw_bdh` arm.
2. Distill `JumpOperator` (jump_size=2) against the teacher's REAL
   trajectories (`bdh_forward_with_trajectory`), with gradient clipping.
3. Evaluate real validation loss AND real wall-clock throughput for:
   `real_depth8` (ground truth), `all_jumps`, `hybrid_2real_Njump`,
   `hybrid_4real_Njump` -- same arm structure as the original prototype,
   now at real scale.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_checkpointed_torch import bdh_variable_depth_forward_checkpointed
from reference.hz0h_bdh_jump_operator_torch import JumpOperator, jump_bdh_forward
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_trajectory_torch import bdh_forward_with_trajectory
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward
from reference.hz0h_bdh_wide_gemm_trainable_torch import bdh_wide_gemm_trainable_forward
from scripts.hz0h_bdh_combined_best_comparison import autocast_context, curriculum_stages, make_optimizer
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import depth_at, lr_at, read_batch


def train_teacher(config: BDHConfig, args, device) -> BDH:
    torch.manual_seed(args.seed)
    model = BDH(config).to(device=device, dtype=torch.float32)
    optimizer = make_optimizer(model.parameters(), args, device)
    steps = -(-args.target_tokens // (args.batch_size * args.sequence_length))
    stages = curriculum_stages(args.target_tokens, config.n_layer)
    epochs = [0]
    tokens = 0

    if args.use_wide_gemm and args.gradient_checkpointing:
        raise RuntimeError(
            "--use-wide-gemm + --gradient-checkpointing is not implemented yet -- real, disclosed "
            "gap, not a silent fallback. bdh_wide_gemm_trainable_forward has no checkpointed variant. "
            "Use one or the other for now."
        )
    if args.use_wide_gemm:
        raw_fn = bdh_wide_gemm_trainable_forward
    else:
        raw_fn = bdh_variable_depth_forward_checkpointed if args.gradient_checkpointing else bdh_variable_depth_forward
    forward_fn = torch.compile(raw_fn, mode=args.compile_mode) if args.compile_training else raw_fn

    started = time.perf_counter()
    with args.data.open() as handle:
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, steps, args.warmup_steps, args.learning_rate)
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            depth = depth_at(tokens, stages)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args, device):
                _, loss = forward_fn(model, idx, depth, target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            tokens += args.batch_size * args.sequence_length
            if args.log_every and (step + 1) % args.log_every == 0:
                now = time.perf_counter()
                rate = tokens / (now - started)
                eta = (steps - step - 1) / max(step + 1, 1) * (now - started)
                print(f"[teacher] step {step+1}/{steps} depth={depth} loss={float(loss):.4f} "
                      f"{rate:.0f} tok/s eta={eta:.0f}s", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[teacher] DONE {tokens} tokens in {elapsed:.0f}s final_loss={float(loss):.4f}", flush=True)
    model.eval()
    return model, elapsed


def train_jump(model: BDH, args, device) -> tuple:
    jump = JumpOperator(d_model=args.n_embd, hidden_mult=args.jump_hidden_mult, jump_size=2).to(device)
    optimizer = make_optimizer(jump.parameters(), args, device)
    epochs = [0]
    starting_depths = [r for r in range(0, args.n_layer - 1, 2)]
    started = time.perf_counter()
    with args.data.open() as handle:
        for step in range(args.jump_steps):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx = data[:, :-1].contiguous()
            with torch.no_grad(), autocast_context(args, device):
                _, _, x_states, _ = bdh_forward_with_trajectory(model, idx, args.n_layer)
            r = random.choice(starting_depths)
            x_r, x_target = x_states[r].detach(), x_states[r + 2].detach()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args, device):
                predicted = jump(x_r)
                state_loss = torch.nn.functional.mse_loss(predicted, x_target)
                B, _, T, D = predicted.shape
                with torch.no_grad():
                    target_logits = x_target.view(B, T, D) @ model.lm_head
                predicted_logits = predicted.view(B, T, D) @ model.lm_head
                logits_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(predicted_logits, dim=-1),
                    torch.nn.functional.softmax(target_logits, dim=-1), reduction="batchmean",
                )
            (state_loss + 0.1 * logits_loss).backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(jump.parameters(), args.grad_clip)
            optimizer.step()
            if args.log_every and (step + 1) % args.log_every == 0:
                now = time.perf_counter()
                rate = (step + 1) / (now - started)
                eta = (args.jump_steps - step - 1) / max(step + 1, 1) * (now - started)
                print(f"[jump] step {step+1}/{args.jump_steps} state_loss={float(state_loss):.4f} "
                      f"logits_loss={float(logits_loss):.4f} {rate:.1f} step/s eta={eta:.0f}s", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[jump] DONE {args.jump_steps} steps in {elapsed:.0f}s "
          f"final_state_loss={float(state_loss):.4f}", flush=True)
    jump.eval()
    return jump, elapsed


def evaluate_arm(model, jump, args, device, real_prefix: int, num_jumps: int) -> dict:
    epochs = [0]
    losses = []
    with args.validation_data.open() as handle, torch.no_grad(), autocast_context(args, device):
        for _ in range(args.eval_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            if jump is None:
                _, loss = bdh_variable_depth_forward(model, idx, real_prefix, target)
            else:
                _, loss = jump_bdh_forward(model, jump, idx, real_prefix, num_jumps, target)
            losses.append(float(loss))

    idx = torch.randint(256, (args.batch_size, args.sequence_length), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), autocast_context(args, device):
        for _ in range(3):
            jump_bdh_forward(model, jump, idx, real_prefix, num_jumps) if jump is not None \
                else bdh_variable_depth_forward(model, idx, real_prefix)
        synchronize(device)
        steps = 10
        started = time.perf_counter()
        for _ in range(steps):
            jump_bdh_forward(model, jump, idx, real_prefix, num_jumps) if jump is not None \
                else bdh_variable_depth_forward(model, idx, real_prefix)
        synchronize(device)
    elapsed = time.perf_counter() - started
    tokens = steps * args.batch_size * args.sequence_length

    result = {
        "validation_loss": sum(losses) / len(losses),
        "real_prefix_iterations": real_prefix,
        "num_jumps": num_jumps,
        "depth_equivalent": real_prefix + num_jumps * 2,
        "tokens_per_second": tokens / elapsed,
    }
    if device.type == "cuda":
        result["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--use-wide-gemm", action="store_true",
                        help="Use reference/hz0h_bdh_wide_gemm_trainable_torch.py's wide-GEMM encoder + "
                             "batched-GEMM encoder_v layout instead of the oracle's broadcasted per-head "
                             "matmuls -- real, prior-measured forward-only wins (1.705x/1.509x) now wired "
                             "for training. Not yet combinable with --gradient-checkpointing (real gap, "
                             "errors loudly rather than silently ignoring the flag).")
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="max-autotune")
    parser.add_argument("--compile-training", action="store_true",
                        help="torch.compile the teacher's training forward pass. See "
                             "scripts/hz0h_bdh_combined_best_comparison.py's --compile-training for the "
                             "full rationale (max-autotune default, real +4.6%%/-96%% memory prior result, "
                             "compile+checkpointing combo untested in this project's history).")
    parser.add_argument("--target-tokens", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--jump-steps", type=int, default=500)
    parser.add_argument("--jump-hidden-mult", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    args = parser.parse_args()

    device = pick_device(args.device)
    random.seed(args.seed)

    config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0)
    print(f"=== training teacher (n_embd={args.n_embd}, mult={args.mult}, plain attention) ===", flush=True)
    model, teacher_seconds = train_teacher(config, args, device)
    teacher_params = sum(p.numel() for p in model.parameters())

    print("=== distilling jump operator against teacher's real trajectories ===", flush=True)
    jump, jump_seconds = train_jump(model, args, device)
    jump_params = sum(p.numel() for p in jump.parameters())

    print("=== evaluating arms ===", flush=True)
    half_depth = args.n_layer // 2
    arms = {
        "real_depth8": evaluate_arm(model, None, args, device, real_prefix=args.n_layer, num_jumps=0),
        "all_jumps": evaluate_arm(model, jump, args, device, real_prefix=0, num_jumps=half_depth),
        "hybrid_2real": evaluate_arm(model, jump, args, device, real_prefix=2, num_jumps=(args.n_layer - 2) // 2),
        "hybrid_4real": evaluate_arm(model, jump, args, device, real_prefix=4, num_jumps=(args.n_layer - 4) // 2),
    }
    baseline = arms["real_depth8"]["validation_loss"]
    baseline_tokps = arms["real_depth8"]["tokens_per_second"]
    for name, arm in arms.items():
        arm["validation_loss_minus_real_depth8"] = arm["validation_loss"] - baseline
        arm["speedup_vs_real_depth8"] = arm["tokens_per_second"] / baseline_tokps
        print(f"[{name}] loss={arm['validation_loss']:.4f} delta={arm['validation_loss_minus_real_depth8']:+.4f} "
              f"depth_eq={arm['depth_equivalent']} tok/s={arm['tokens_per_second']:.0f} "
              f"speedup={arm['speedup_vs_real_depth8']:.2f}x "
              f"peak_mem={arm.get('peak_memory_gb', 'n/a')}", flush=True)

    report = {
        "device": str(device),
        "shape": {"n_embd": args.n_embd, "mult": args.mult, "n_layer": args.n_layer, "n_head": args.n_head,
                  "dtype": args.dtype, "optimizer": args.optimizer, "target_tokens": args.target_tokens},
        "teacher_params": teacher_params, "jump_params": jump_params,
        "teacher_training_seconds": teacher_seconds, "jump_training_seconds": jump_seconds,
        "arms": arms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
