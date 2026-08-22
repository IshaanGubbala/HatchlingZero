#!/usr/bin/env python3
"""Fresh-process, full-training-step CUDA benchmark: packed encoder alone
(+4.19% over the original checkpointed forward) vs packed encoder +
symmetric attention backward combined (each independently validated;
this is the integration the Highest-Value Ideas review's recommended
order held off combining until now).
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

from reference.hz0h_bdh_packed_encoder_torch import PackedEncoderBDH, bdh_packed_encoder_forward_checkpointed
from reference.hz0h_bdh_packed_encoder_symmetric_torch import bdh_packed_encoder_symmetric_forward_checkpointed
from reference.hz0h_bdh_torch import BDHConfig

ARMS = ("packed", "packed_symmetric")


def make_model_and_batch(args, device):
    torch.manual_seed(args.seed)
    config = BDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.multiplier, vocab_size=args.vocab_size, dropout=0.0,
    )
    model = PackedEncoderBDH(config).to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    idx = torch.randint(args.vocab_size, (args.batch_size, args.sequence_length), device=device, generator=generator)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.sequence_length), device=device, generator=generator)
    return model, idx, targets


def one_step(arm, model, idx, targets, optimizer, checkpoint_segment_size, grad_clip):
    optimizer.zero_grad(set_to_none=True)
    forward_fn = bdh_packed_encoder_forward_checkpointed if arm == "packed" else bdh_packed_encoder_symmetric_forward_checkpointed
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = forward_fn(model, idx, model.config.n_layer, targets, checkpoint_segment_size=checkpoint_segment_size)
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss


def run_child(args):
    device = torch.device("cuda")
    model, idx, targets = make_model_and_batch(args, device)
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
    packed_model, idx, targets = make_model_and_batch(args, device)
    combined_model, _, _ = make_model_and_batch(args, device)

    packed_opt = torch.optim.AdamW(packed_model.parameters(), lr=3e-4)
    combined_opt = torch.optim.AdamW(combined_model.parameters(), lr=3e-4)

    packed_loss = one_step("packed", packed_model, idx, targets, packed_opt, args.checkpoint_segment_size, args.grad_clip)
    combined_loss = one_step("packed_symmetric", combined_model, idx, targets, combined_opt, args.checkpoint_segment_size, args.grad_clip)

    worst = 0.0
    for name, p in packed_model.named_parameters():
        pc = dict(combined_model.named_parameters())[name]
        worst = max(worst, (p.detach() - pc.detach()).abs().max().item())

    return {
        "loss_max_abs_diff": float((packed_loss - combined_loss).abs()),
        "post_adamw_step_param_max_abs_diff": worst,
        "finite": bool(torch.isfinite(packed_loss) and torch.isfinite(combined_loss)),
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
    with tempfile.TemporaryDirectory(prefix="hz0h_packed_symmetric_full_step_") as directory:
        for arm in ARMS:
            child_out = Path(directory) / f"{arm}.json"
            subprocess.run(child_command(args, arm, child_out), check=True)
            results[arm] = json.loads(child_out.read_text(encoding="utf-8"))
    packed = results["packed"]
    combined = results["packed_symmetric"]
    report = {
        "experiment_id": "bdh_packed_encoder_symmetric_full_step_v1",
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
        "algorithmic_difference": (
            "packed encoder (native (D, nh*N) parameter) + symmetric "
            "attention backward ((dS+dS.T)@Q identity), combined -- both "
            "independently validated as real wins, this is the integration"
        ),
        "fresh_subprocess_per_arm": True,
        "one_step_parity": parity,
        "arms": results,
        "combined_over_packed_throughput": combined["tokens_per_second_mean"] / packed["tokens_per_second_mean"],
        "combined_over_packed_allocated_memory": combined["peak_memory_allocated_bytes"] / packed["peak_memory_allocated_bytes"],
        "combined_over_packed_reserved_memory": combined["peak_memory_reserved_bytes"] / packed["peak_memory_reserved_bytes"],
        "stop_condition": "close if one-step parity fails or combined does not beat packed alone on throughput",
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
    parser.add_argument("--arm", choices=list(ARMS), default="packed", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.child:
        run_child(parsed)
    else:
        run_parent(parsed)
