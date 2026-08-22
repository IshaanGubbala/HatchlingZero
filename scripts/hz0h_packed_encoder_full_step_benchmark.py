#!/usr/bin/env python3
"""Fresh-process, full-training-step CUDA benchmark: existing checkpointed
wide-GEMM (repacks encoder 16x/step at n_layer=8) vs cached_encoder (1x/step,
already measured +1.94%) vs packed (0x/step -- model.encoder_packed IS the
trainable parameter, nothing ever repacked).

Same methodology as
hz0h_wide_gemm_checkpointed_cached_encoder_full_step_benchmark.py, extended
to the third arm.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_wide_gemm_checkpointed_torch import bdh_wide_gemm_forward_checkpointed
from reference.hz0h_bdh_wide_gemm_checkpointed_cached_encoder_torch import (
    bdh_wide_gemm_forward_checkpointed_cached_encoder,
)
from reference.hz0h_bdh_packed_encoder_torch import PackedEncoderBDH, bdh_packed_encoder_forward_checkpointed

ARMS = ("existing", "cached_encoder", "packed")


def make_model_and_batch(args, device, arm):
    torch.manual_seed(args.seed)
    config = BDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.multiplier, vocab_size=args.vocab_size, dropout=0.0,
    )
    model_cls = PackedEncoderBDH if arm == "packed" else BDH
    model = model_cls(config).to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    idx = torch.randint(args.vocab_size, (args.batch_size, args.sequence_length), device=device, generator=generator)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.sequence_length), device=device, generator=generator)
    return model, idx, targets


def one_step(arm, model, idx, targets, optimizer, checkpoint_segment_size, grad_clip):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if arm == "existing":
            _, loss = bdh_wide_gemm_forward_checkpointed(model, idx, model.config.n_layer, targets, checkpoint_segment_size=checkpoint_segment_size)
        elif arm == "cached_encoder":
            _, loss = bdh_wide_gemm_forward_checkpointed_cached_encoder(model, idx, model.config.n_layer, targets, checkpoint_segment_size=checkpoint_segment_size)
        else:
            _, loss = bdh_packed_encoder_forward_checkpointed(model, idx, model.config.n_layer, targets, checkpoint_segment_size=checkpoint_segment_size)
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss


def run_child(args):
    device = torch.device("cuda")
    model, idx, targets = make_model_and_batch(args, device, args.arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for _ in range(args.warmup):
        one_step(args.arm, model, idx, targets, optimizer, args.checkpoint_segment_size, args.grad_clip)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    seconds_per_step_trials = []
    loss_value = None
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.steps):
            loss_value = float(one_step(args.arm, model, idx, targets, optimizer, args.checkpoint_segment_size, args.grad_clip))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        seconds_per_step_trials.append(elapsed / args.steps)

    mean = sum(seconds_per_step_trials) / len(seconds_per_step_trials)
    variance = sum((t - mean) ** 2 for t in seconds_per_step_trials) / len(seconds_per_step_trials)
    result = {
        "arm": args.arm,
        "seconds_per_step_trials": seconds_per_step_trials,
        "seconds_per_step_mean": mean,
        "seconds_per_step_stdev": variance ** 0.5,
        "tokens_per_second_mean": args.batch_size * args.sequence_length / mean,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "final_loss": loss_value,
        "finite": bool(torch.isfinite(torch.tensor(loss_value))),
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


def verify_parity(args):
    device = torch.device("cuda")
    existing_model, idx, targets = make_model_and_batch(args, device, "existing")
    cached_model, _, _ = make_model_and_batch(args, device, "cached_encoder")
    packed_model, _, _ = make_model_and_batch(args, device, "packed")
    cached_model.load_state_dict(existing_model.state_dict())
    # packed_model was constructed with the same seed, so its weights are
    # already bit-identical to existing_model's (just relabeled storage) --
    # see test_hz0h_bdh_packed_encoder_torch.py for the proof this holds.

    existing_opt = torch.optim.AdamW(existing_model.parameters(), lr=3e-4)
    cached_opt = torch.optim.AdamW(cached_model.parameters(), lr=3e-4)
    packed_opt = torch.optim.AdamW(packed_model.parameters(), lr=3e-4)

    existing_loss = one_step("existing", existing_model, idx, targets, existing_opt, args.checkpoint_segment_size, args.grad_clip)
    cached_loss = one_step("cached_encoder", cached_model, idx, targets, cached_opt, args.checkpoint_segment_size, args.grad_clip)
    packed_loss = one_step("packed", packed_model, idx, targets, packed_opt, args.checkpoint_segment_size, args.grad_clip)

    def max_param_diff(model_a, model_b, skip_name=None):
        worst = 0.0
        params_a = dict(model_a.named_parameters())
        params_b = dict(model_b.named_parameters())
        for name, p_a in params_a.items():
            if name == skip_name:
                continue
            worst = max(worst, (p_a.detach() - params_b[name].detach()).abs().max().item())
        return worst

    nh, D, N = existing_model.encoder.shape
    from reference.hz0h_bdh_packed_encoder_torch import unpack_encoder_view
    packed_encoder_unpacked = unpack_encoder_view(packed_model.encoder_packed.detach(), nh, N)
    encoder_diff = (existing_model.encoder.detach() - packed_encoder_unpacked).abs().max().item()

    return {
        "existing_vs_cached_loss_max_abs_diff": float((existing_loss - cached_loss).abs()),
        "existing_vs_packed_loss_max_abs_diff": float((existing_loss - packed_loss).abs()),
        "existing_vs_cached_param_max_abs_diff": max_param_diff(existing_model, cached_model),
        "existing_vs_packed_non_encoder_param_max_abs_diff": max_param_diff(existing_model, packed_model, skip_name="encoder"),
        "existing_vs_packed_encoder_max_abs_diff_through_unpack": encoder_diff,
        "finite": bool(torch.isfinite(existing_loss) and torch.isfinite(cached_loss) and torch.isfinite(packed_loss)),
    }


def child_command(args, arm, out):
    return [
        sys.executable, str(Path(__file__).resolve()),
        "--child", "--arm", arm, "--out", str(out),
        "--batch-size", str(args.batch_size),
        "--sequence-length", str(args.sequence_length),
        "--n-embd", str(args.n_embd),
        "--n-head", str(args.n_head),
        "--multiplier", str(args.multiplier),
        "--n-layer", str(args.n_layer),
        "--vocab-size", str(args.vocab_size),
        "--checkpoint-segment-size", str(args.checkpoint_segment_size),
        "--grad-clip", str(args.grad_clip),
        "--warmup", str(args.warmup),
        "--steps", str(args.steps),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
    ]


def run_parent(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    parity = verify_parity(args)
    results = {}
    with tempfile.TemporaryDirectory(prefix="hz0h_packed_encoder_full_step_") as directory:
        for arm in ARMS:
            child_out = Path(directory) / f"{arm}.json"
            subprocess.run(child_command(args, arm, child_out), check=True)
            results[arm] = json.loads(child_out.read_text(encoding="utf-8"))
    existing = results["existing"]
    cached = results["cached_encoder"]
    packed = results["packed"]
    report = {
        "experiment_id": "bdh_packed_encoder_full_step_v1",
        "scope": "complete production training step: forward, backward, grad clipping, one AdamW step",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "config": {
            "batch_size": args.batch_size, "sequence_length": args.sequence_length,
            "n_embd": args.n_embd, "n_head": args.n_head, "multiplier": args.multiplier,
            "n_layer": args.n_layer, "vocab_size": args.vocab_size,
            "checkpoint_segment_size": args.checkpoint_segment_size, "grad_clip": args.grad_clip,
        },
        "fresh_subprocess_per_arm": True,
        "one_step_parity": parity,
        "arms": results,
        "cached_over_existing_throughput": cached["tokens_per_second_mean"] / existing["tokens_per_second_mean"],
        "packed_over_existing_throughput": packed["tokens_per_second_mean"] / existing["tokens_per_second_mean"],
        "packed_over_cached_throughput": packed["tokens_per_second_mean"] / cached["tokens_per_second_mean"],
        "packed_over_existing_allocated_memory": packed["peak_memory_allocated_bytes"] / existing["peak_memory_allocated_bytes"],
        "packed_over_existing_reserved_memory": packed["peak_memory_reserved_bytes"] / existing["peak_memory_reserved_bytes"],
        "packed_over_cached_reserved_memory": packed["peak_memory_reserved_bytes"] / cached["peak_memory_reserved_bytes"],
        "stop_condition": "close if one-step parity fails, or packed does not beat cached_encoder on throughput or memory",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--multiplier", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--checkpoint-segment-size", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=list(ARMS), default="existing", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.child:
        run_child(parsed)
    else:
        run_parent(parsed)
