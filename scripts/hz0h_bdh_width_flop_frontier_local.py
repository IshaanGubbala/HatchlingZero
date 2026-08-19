#!/usr/bin/env python3
"""Local (MPS/CPU) scaled-down version of the BDH latent-width frontier
sweep, for when the CUDA machine is unavailable.

Same real question as `scripts/hz0h_bdh_width_flop_frontier.py`: BDH's
three wide projections are 82.8% of per-level FLOPs and scale LINEARLY
with `mlp_internal_dim_multiplier`, so is the canonical 32x actually
load-bearing for quality, or is it inherited from the paper and never
validated?

**Real, disclosed limits of running this locally -- read before trusting
any number it produces:**

- This is a SCALED-DOWN model (smaller `n_embd`/`n_layer`/sequence
  length) on a SMALLER token budget than the real 25M-token comparison.
  Its absolute validation losses are NOT comparable to the established
  reference numbers (dense BDH 1.3848, matched Transformer 1.5141) and
  must never be quoted alongside them as if they were.
- What it CAN honestly provide is the SHAPE of the width/quality
  tradeoff at reduced scale -- whether quality falls off a cliff, decays
  gently, or is flat as width shrinks. That shape is the actual decision
  input for whether the full CUDA sweep is worth running and where to
  concentrate it.
- This project has a real, documented instance of a short/small probe
  giving a misleading answer that reversed at full scale
  (`docs/restart/hz0h_factorized_curriculum_full_comparison_results.md`).
  The depth curriculum is therefore still applied here, and any finding
  from this script is explicitly preliminary until the CUDA run confirms
  it.
- fp32 on MPS (not bf16): MPS bf16 support is uneven and BDH's own
  `Attention` asserts fp32 RoPE frequencies. Precision therefore differs
  from the CUDA runs by design, another reason absolute numbers here are
  not cross-comparable.

Recipe fidelity: imports `read_batch`, `lr_at`, `depth_at`,
`parse_stages`, `loss_for` and `evaluate` directly from
`scripts/hz0h_factorized_curriculum_full_comparison.py` so the data
pipeline, schedule, curriculum and eval path cannot drift from the
established comparison. Only that script's CUDA-specific training loop
(energy sampler, `torch.cuda` memory stats, hardcoded device) is
replaced here with a device-flexible equivalent.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM
from reference.hz0h_bdh_torch import BDH, BDHConfig
from scripts.hz0h_bdh_cost_breakdown_profile import analytic_flops, matched_transformer_flops
from scripts.hz0h_factorized_curriculum_full_comparison import (
    depth_at,
    evaluate,
    loss_for,
    lr_at,
    parse_stages,
    read_batch,
)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def train_local(name: str, model, kind: str, args, stages, device: torch.device) -> dict:
    """Device-flexible equivalent of the curriculum harness's own `train`,
    minus the CUDA-only energy/memory instrumentation."""
    steps = math.ceil(args.target_tokens / (args.batch_size * args.sequence_length))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    epochs = [0]
    tokens = 0
    history = []
    best = float("inf")
    started = time.perf_counter()
    with args.data.open() as handle:
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, steps, args.warmup_steps, args.learning_rate)
            batch = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            depth = depth_at(tokens, stages) if kind != "transformer" else args.n_layer
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model, kind, idx, target, depth)
            loss.backward()
            optimizer.step()
            tokens += args.batch_size * args.sequence_length
            if (step + 1) % args.eval_every == 0 or step + 1 == steps:
                eval_depth = stages[-1][1] if (step + 1 == steps and kind != "transformer") else depth
                validation = evaluate(
                    model, kind, args.validation_data, args.validation_batch_size,
                    args.sequence_length, device, args.eval_batches, eval_depth,
                )
                best = min(best, validation)
                history.append({
                    "step": step + 1, "tokens_seen": tokens, "depth": depth,
                    "validation_loss": validation,
                })
    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "name": name,
        "kind": kind,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "tokens_seen": tokens,
        "steps": steps,
        "epochs": epochs[0],
        "best_validation_loss": best,
        "final_validation_loss": history[-1]["validation_loss"] if history else None,
        "validation_history": history,
        "training_seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-tokens", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--multipliers", default="32,16,8,4")
    parser.add_argument("--skip-transformer", action="store_true")
    parser.add_argument("--curriculum-stages", default=None,
                        help="Defaults to a scaled 2->3->4 schedule proportional to --target-tokens.")
    args = parser.parse_args()

    device = pick_device(args.device)
    if args.curriculum_stages is None:
        # Real, scaled equivalent of the canonical 2->4->6->8 schedule,
        # proportioned to this run's own token budget and n_layer.
        quarter = args.target_tokens // 4
        depths = sorted({max(2, round(args.n_layer * fraction)) for fraction in (0.5, 0.75, 1.0)})
        boundaries = [quarter * 2, quarter * 3, args.target_tokens][-len(depths):]
        args.curriculum_stages = ",".join(f"{b}:{d}" for b, d in zip(boundaries, depths))
    stages = parse_stages(args.curriculum_stages)
    multipliers = [int(value) for value in args.multipliers.split(",")]

    arms = {}
    for multiplier in multipliers:
        torch.manual_seed(args.seed)
        config = BDHConfig(
            n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
            mlp_internal_dim_multiplier=multiplier, vocab_size=256, dropout=0.0,
        )
        model = BDH(config).to(device=device, dtype=torch.float32)
        name = f"bdh_mult_{multiplier}"
        print(f"[{name}] starting on {device} ...", flush=True)
        result = train_local(name, model, "dense", args, stages, device)
        latent_width = args.n_embd * multiplier // args.n_head
        level_flops = analytic_flops(
            args.batch_size, args.sequence_length, args.n_embd, args.n_head, latent_width,
        )
        result["latent_width_per_head"] = latent_width
        result["mlp_internal_dim_multiplier"] = multiplier
        result["forward_flops_at_full_depth"] = sum(level_flops.values()) * args.n_layer
        arms[name] = result
        print(f"[{name}] best_val={result['best_validation_loss']:.4f} "
              f"params={result['parameter_count']/1e6:.2f}M "
              f"seconds={result['training_seconds']:.0f}", flush=True)
        del model

    if not args.skip_transformer:
        torch.manual_seed(args.seed)
        transformer_config = MatchedTransformerConfig({
            "vocab_size": 256, "d_model": args.n_embd, "num_layers": args.n_layer,
            "num_heads": args.n_head, "head_dim": args.n_embd // args.n_head,
            "d_ff": args.n_embd * 4, "use_rope": True,
        })
        transformer = MatchedTransformerLM(transformer_config).to(device=device, dtype=torch.float32)
        print(f"[matched_transformer] starting on {device} ...", flush=True)
        result = train_local("matched_transformer", transformer, "transformer", args, stages, device)
        result["forward_flops_at_full_depth"] = matched_transformer_flops(
            args.batch_size, args.sequence_length, args.n_embd, args.n_layer,
            args.n_head, args.n_embd // args.n_head, args.n_embd * 4,
        )["whole_model"]
        arms["matched_transformer"] = result
        print(f"[matched_transformer] best_val={result['best_validation_loss']:.4f} "
              f"params={result['parameter_count']/1e6:.2f}M "
              f"seconds={result['training_seconds']:.0f}", flush=True)
        del transformer

    baseline = arms.get(f"bdh_mult_{multipliers[0]}")
    for arm in arms.values():
        if baseline and arm.get("forward_flops_at_full_depth"):
            arm["flops_vs_widest_bdh"] = arm["forward_flops_at_full_depth"] / baseline["forward_flops_at_full_depth"]
            arm["validation_loss_minus_widest_bdh"] = arm["best_validation_loss"] - baseline["best_validation_loss"]

    report = {
        "device": str(device),
        "dtype": "float32",
        "scaled_down_local_run": True,
        "not_comparable_to_cuda_reference_numbers": (
            "Absolute losses here are NOT comparable to the 25M-token CUDA comparison "
            "(dense BDH 1.3848, matched Transformer 1.5141) -- smaller model, smaller "
            "token budget, fp32 instead of bf16. Only the SHAPE of the width/quality "
            "tradeoff is the real signal from this run."
        ),
        "curriculum_stages": stages,
        "target_tokens": args.target_tokens,
        "shape": {
            "n_embd": args.n_embd, "n_layer": args.n_layer, "n_head": args.n_head,
            "batch_size": args.batch_size, "sequence_length": args.sequence_length,
        },
        "multipliers_swept": multipliers,
        "arms": arms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: {
        "params": arm["parameter_count"],
        "best_validation_loss": arm["best_validation_loss"],
        "flops_vs_widest_bdh": arm.get("flops_vs_widest_bdh"),
        "training_seconds": round(arm["training_seconds"], 1),
    } for name, arm in arms.items()}, indent=2))


if __name__ == "__main__":
    main()
