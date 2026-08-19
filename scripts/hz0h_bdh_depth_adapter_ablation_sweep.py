#!/usr/bin/env python3
"""Real training sweep testing whether AdapterDepthBDH's shared-base +
low-rank-per-group correction recovers most of full-capacity untying's
quality win at a small fraction of its parameter cost.

Part 4c (`docs/restart/hz0h_inherited_choices_audit_results.md`) found
that giving every recurrent group its OWN full-size encoder/encoder_v/
decoder (`untied_full_capacity`) matches or beats tied quality at real
depth (`n_layer=8`), but costs up to 7.72x the parameters at groups=8.
This sweep asks: does a MUCH smaller per-group correction (rank-`rank`
low-rank delta on a shared base) get most of that win back?

Arms, all at `groups=8` (the most expensive full-untying case, so the
biggest potential saving): `tied_baseline` (real oracle), then
`adapter_r{rank}` for a few rank values, each compared against BOTH
`tied_baseline` (does it beat tying at all?) and the ALREADY-KNOWN
`untied_full_capacity` number from Part 4c (does it recover that win at
much lower param cost?).

Real, disclosed limits (same harness as the depth-untying sweeps this
reuses): scaled-down or real-depth local run depending on --n-layer,
fp32 on MPS. Absolute losses are NOT comparable to the 25M-token CUDA
reference numbers.
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

from reference.hz0h_bdh_depth_adapter_torch import AdapterDepthBDH
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import depth_at, lr_at, parse_stages, read_batch


def evaluate_tied(model, path, batch_size, sequence_length, device, batches, depth) -> float:
    model.eval()
    epochs = [0]
    losses = []
    with path.open() as handle, torch.no_grad():
        for _ in range(batches):
            data = read_batch(handle, batch_size, sequence_length, device, epochs)
            _, loss = bdh_variable_depth_forward(model, data[:, :-1].contiguous(), depth, data[:, 1:].contiguous())
            losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def evaluate_adapter(model, path, batch_size, sequence_length, device, batches, depth) -> float:
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
                evaluator = evaluate_tied if tied else evaluate_adapter
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
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--tied-multiplier", type=int, default=16)
    parser.add_argument("--groups", type=int, default=8,
                        help="Fixed at 8 by default -- the most expensive full-untying case "
                             "(7.72x params in Part 4c), so the biggest potential saving.")
    parser.add_argument("--ranks", default="4,8,16")
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
    ranks = [int(r) for r in args.ranks.split(",") if r.strip()]

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

    for rank in ranks:
        arm_name = f"adapter_g{args.groups}_r{rank}"
        torch.manual_seed(args.seed)
        adapter_config = BDHConfig(
            n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
            mlp_internal_dim_multiplier=args.tied_multiplier, vocab_size=256, dropout=0.0,
        )
        adapter_model = AdapterDepthBDH(
            adapter_config, depth=max_depth, groups=args.groups, rank=rank,
        ).to(device=device, dtype=torch.float32)
        print(f"[{arm_name}] starting on {device} (groups={args.groups} rank={rank}) ...", flush=True)
        result = train_arm(arm_name, adapter_model, args, stages, device, tied=False)
        result["groups"] = args.groups
        result["rank"] = rank
        result["adapter_extra_params"] = adapter_model.adapter_parameter_count()
        arms[arm_name] = result
        print(f"[{arm_name}] best_val={result['best_validation_loss']:.4f} "
              f"params={result['parameter_count']/1e6:.2f}M "
              f"adapter_extra={result['adapter_extra_params']/1e3:.1f}K seconds={result['training_seconds']:.0f}", flush=True)
        del adapter_model

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
            "(dense BDH 1.3848, matched Transformer 1.5141)."
        ),
        "curriculum_stages": stages,
        "target_tokens": args.target_tokens,
        "shape": {
            "n_embd": args.n_embd, "n_layer": args.n_layer, "n_head": args.n_head,
            "tied_multiplier": args.tied_multiplier, "groups": args.groups, "ranks_swept": ranks,
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
        "params_vs_tied_baseline": round(a["params_vs_tied_baseline"], 4),
    } for n, a in arms.items()}, indent=2))


if __name__ == "__main__":
    main()
