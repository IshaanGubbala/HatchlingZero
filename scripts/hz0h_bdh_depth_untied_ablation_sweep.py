#!/usr/bin/env python3
"""Real training sweep asking the one question this project's audit never
asked: does BDH's weight tying across recurrent depth actually help, or
is it just inherited?

Three arms, same data/curriculum/seed, only the tying structure differs:

- `tied_baseline`: real oracle BDH, `mlp_internal_dim_multiplier=16` (the
  near-optimal width found by the width sweep) -- one encoder/encoder_v/
  decoder reused at every recurrent level.
- `untied_budget_matched`: `DepthUntiedBDH`, per-level multiplier =
  `budget_matched_multiplier(16, n_layer)` so the untied TOTAL param
  count across all levels is approximately equal to the tied baseline's.
  Isolates tying itself, controlled for capacity.
- `untied_full_capacity`: `DepthUntiedBDH`, per-level multiplier=16 (same
  as tied baseline PER LEVEL) -- `n_layer`x more encoder/encoder_v/
  decoder params than the tied baseline. Real, disclosed confound: any
  win here could be capacity, not tying, which is exactly why the
  budget-matched arm exists as the controlled comparison.

Real, disclosed limits (same harness as the width/primitive sweeps this
reuses): scaled-down model, reduced token budget, fp32 on MPS. Absolute
losses are NOT comparable to the 25M-token CUDA reference numbers.
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

from reference.hz0h_bdh_depth_untied_torch import DepthUntiedBDH, budget_matched_multiplier
from reference.hz0h_bdh_torch import BDH, BDHConfig
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import depth_at, lr_at, parse_stages, read_batch


def evaluate_tied(model, path, batch_size, sequence_length, device, batches, depth) -> float:
    model.eval()
    epochs = [0]
    losses = []
    with path.open() as handle, torch.no_grad():
        for _ in range(batches):
            data = read_batch(handle, batch_size, sequence_length, device, epochs)
            from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward
            _, loss = bdh_variable_depth_forward(model, data[:, :-1].contiguous(), depth, data[:, 1:].contiguous())
            losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def evaluate_untied(model, path, batch_size, sequence_length, device, batches, depth) -> float:
    model.eval()
    epochs = [0]
    losses = []
    with path.open() as handle, torch.no_grad():
        for _ in range(batches):
            data = read_batch(handle, batch_size, sequence_length, device, epochs)
            _, loss = model(data[:, :-1].contiguous(), data[:, 1:].contiguous(), depth=depth)
            losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def train_arm(name, model, args, stages, device, tied: bool) -> dict:
    steps = math.ceil(args.target_tokens / (args.batch_size * args.sequence_length))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    epochs = [0]
    tokens = 0
    history = []
    best = float("inf")
    started = time.perf_counter()
    from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward
    with args.data.open() as handle:
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, steps, args.warmup_steps, args.learning_rate)
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            depth = depth_at(tokens, stages)
            optimizer.zero_grad(set_to_none=True)
            if tied:
                _, loss = bdh_variable_depth_forward(model, data[:, :-1].contiguous(), depth, data[:, 1:].contiguous())
            else:
                _, loss = model(data[:, :-1].contiguous(), data[:, 1:].contiguous(), depth=depth)
            loss.backward()
            optimizer.step()
            tokens += args.batch_size * args.sequence_length
            if (step + 1) % args.eval_every == 0 or step + 1 == steps:
                eval_depth = stages[-1][1] if step + 1 == steps else depth
                evaluator = evaluate_tied if tied else evaluate_untied
                validation = evaluator(
                    model, args.validation_data, args.batch_size,
                    args.sequence_length, device, args.eval_batches, eval_depth,
                )
                best = min(best, validation)
                history.append({"step": step + 1, "depth": depth, "validation_loss": validation})
    synchronize(device)
    return {
        "name": name,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "best_validation_loss": best,
        "final_validation_loss": history[-1]["validation_loss"] if history else None,
        "validation_history": history,
        "training_seconds": time.perf_counter() - started,
        "tokens_seen": tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-tokens", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--tied-multiplier", type=int, default=16,
                        help="Near-optimal width from the width sweep -- used for both the tied "
                             "baseline and the untied full-capacity arm's PER-LEVEL multiplier.")
    parser.add_argument("--curriculum-stages", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    if args.curriculum_stages is None:
        quarter = args.target_tokens // 4
        depths = sorted({max(2, round(args.n_layer * f)) for f in (0.5, 0.75, 1.0)})
        boundaries = [quarter * 2, quarter * 3, args.target_tokens][-len(depths):]
        args.curriculum_stages = ",".join(f"{b}:{d}" for b, d in zip(boundaries, depths))
    stages = parse_stages(args.curriculum_stages)
    max_depth = stages[-1][1]
    matched_multiplier = budget_matched_multiplier(args.tied_multiplier, max_depth)

    arms = {}

    torch.manual_seed(args.seed)
    tied_config = BDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.tied_multiplier, vocab_size=256, dropout=0.0,
    )
    tied_model = BDH(tied_config).to(device=device, dtype=torch.float32)
    print(f"[tied_baseline] starting on {device} ...", flush=True)
    result = train_arm("tied_baseline", tied_model, args, stages, device, tied=True)
    arms["tied_baseline"] = result
    print(f"[tied_baseline] best_val={result['best_validation_loss']:.4f} "
          f"params={result['parameter_count']/1e6:.2f}M seconds={result['training_seconds']:.0f}", flush=True)
    del tied_model

    for arm_name, multiplier in (
        ("untied_budget_matched", matched_multiplier),
        ("untied_full_capacity", args.tied_multiplier),
    ):
        torch.manual_seed(args.seed)
        untied_config = BDHConfig(
            n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
            mlp_internal_dim_multiplier=multiplier, vocab_size=256, dropout=0.0,
        )
        untied_model = DepthUntiedBDH(untied_config, depth=max_depth).to(device=device, dtype=torch.float32)
        print(f"[{arm_name}] starting on {device} (per-level mult={multiplier}) ...", flush=True)
        result = train_arm(arm_name, untied_model, args, stages, device, tied=False)
        result["per_level_multiplier"] = multiplier
        arms[arm_name] = result
        print(f"[{arm_name}] best_val={result['best_validation_loss']:.4f} "
              f"params={result['parameter_count']/1e6:.2f}M seconds={result['training_seconds']:.0f}", flush=True)
        del untied_model

    baseline_loss = arms["tied_baseline"]["best_validation_loss"]
    baseline_params = arms["tied_baseline"]["parameter_count"]
    for arm in arms.values():
        arm["validation_loss_minus_tied_baseline"] = arm["best_validation_loss"] - baseline_loss
        arm["params_vs_tied_baseline"] = arm["parameter_count"] / baseline_params

    report = {
        "device": str(device),
        "dtype": "float32",
        "scaled_down_local_run": True,
        "not_comparable_to_cuda_reference_numbers": (
            "Absolute losses are NOT comparable to the 25M-token CUDA numbers "
            "(dense BDH 1.3848, matched Transformer 1.5141). Only DIRECTION and "
            "rough magnitude of the tying effect is the real signal here."
        ),
        "tied_baseline_uses_real_oracle": (
            "tied_baseline trains the real BDH class via bdh_variable_depth_forward, "
            "not DepthUntiedBDH's tied special case -- the untied module's equivalence "
            "to the oracle is separately proven by "
            "tests/reference/test_hz0h_bdh_depth_untied_torch.py"
        ),
        "budget_matching_rule": (
            f"untied_budget_matched per-level multiplier = "
            f"budget_matched_multiplier({args.tied_multiplier}, depth={max_depth}) = {matched_multiplier}, "
            f"so its TOTAL encoder/encoder_v/decoder params across {max_depth} levels approximate "
            f"tied_baseline's single set."
        ),
        "curriculum_stages": stages,
        "target_tokens": args.target_tokens,
        "shape": {
            "n_embd": args.n_embd, "n_layer": args.n_layer, "n_head": args.n_head,
            "tied_multiplier": args.tied_multiplier, "matched_multiplier": matched_multiplier,
            "batch_size": args.batch_size, "sequence_length": args.sequence_length,
        },
        "seed": args.seed,
        "arms": arms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({n: {
        "best_validation_loss": round(a["best_validation_loss"], 4),
        "vs_tied_baseline": round(a["validation_loss_minus_tied_baseline"], 4),
        "params_vs_tied_baseline": round(a["params_vs_tied_baseline"], 3),
    } for n, a in arms.items()}, indent=2))


if __name__ == "__main__":
    main()
