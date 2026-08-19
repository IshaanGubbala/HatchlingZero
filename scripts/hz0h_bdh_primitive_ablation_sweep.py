#!/usr/bin/env python3
"""Real training sweep over exact BDH's INHERITED attention primitives --
the constants copied verbatim from upstream and never questioned.

See `reference/hz0h_bdh_primitive_ablations_torch.py` for what each knob
is and why it is worth testing. Every arm here trains the SAME model
architecture on the SAME real data with the SAME curriculum and seed;
only one inherited primitive changes per arm, so any quality difference
is attributable to that primitive rather than to a confound.

The baseline arm is the oracle's own configuration, proven bit-for-bit
identical to `BDH.forward` by
`tests/reference/test_hz0h_bdh_primitive_ablations_torch.py` -- that
equivalence is what makes every other arm's delta meaningful.

Real, disclosed limits (same as the local width sweep this reuses the
harness from): scaled-down model, reduced token budget, fp32 on MPS.
Absolute losses are NOT comparable to the 25M-token CUDA reference
numbers. What this run can honestly establish is DIRECTION and rough
magnitude -- which inherited primitives look actively suboptimal enough
to be worth a full CUDA confirmation, and which are already fine.
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

from reference.hz0h_bdh_primitive_ablations_torch import ablated_bdh_forward, build_rope_freqs
from reference.hz0h_bdh_torch import BDH, BDHConfig
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import depth_at, lr_at, parse_stages, read_batch


def evaluate_variant(model, variant, path, batch_size, sequence_length, device, batches, depth) -> float:
    model.eval()
    epochs = [0]
    losses = []
    with path.open() as handle, torch.no_grad():
        for _ in range(batches):
            data = read_batch(handle, batch_size, sequence_length, device, epochs)
            _, loss = ablated_bdh_forward(
                model, data[:, :-1].contiguous(), depth, data[:, 1:].contiguous(), **variant,
            )
            losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def train_variant(name, model, variant, args, stages, device) -> dict:
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
            _, loss = ablated_bdh_forward(
                model, data[:, :-1].contiguous(), depth, data[:, 1:].contiguous(), **variant,
            )
            loss.backward()
            optimizer.step()
            tokens += args.batch_size * args.sequence_length
            if (step + 1) % args.eval_every == 0 or step + 1 == steps:
                eval_depth = stages[-1][1] if step + 1 == steps else depth
                validation = evaluate_variant(
                    model, variant, args.validation_data, args.batch_size,
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
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=16,
                        help="Defaults to 16, the near-optimal width found by the width sweep -- "
                             "2x cheaper than canonical 32 at almost no quality cost, so more "
                             "ablation arms fit in the same budget.")
    parser.add_argument("--curriculum-stages", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    if args.curriculum_stages is None:
        quarter = args.target_tokens // 4
        depths = sorted({max(2, round(args.n_layer * f)) for f in (0.5, 0.75, 1.0)})
        boundaries = [quarter * 2, quarter * 3, args.target_tokens][-len(depths):]
        args.curriculum_stages = ",".join(f"{b}:{d}" for b, d in zip(boundaries, depths))
    stages = parse_stages(args.curriculum_stages)

    def build_model() -> BDH:
        torch.manual_seed(args.seed)
        config = BDHConfig(
            n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
            mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
            vocab_size=256, dropout=0.0,
        )
        return BDH(config).to(device=device, dtype=torch.float32)

    probe = build_model()
    variants = {
        # Baseline: bit-for-bit the real oracle (test-proven).
        "baseline_upstream": {},
        # Inherited theta=2**16 vs the standard RoPE value, and one step further out.
        "rope_theta_10000": {"freqs": build_rope_freqs(probe, 10_000)},
        "rope_theta_1048576": {"freqs": build_rope_freqs(probe, 2 ** 20)},
        # Upstream forbids a token attending to itself; standard causal masks allow it.
        "self_inclusive_mask": {"mask_diagonal": 0},
        # Kept as a real CONTROL, not a candidate: without a softmax, constant
        # score scaling is near-inert because `yKV = ln(scores @ x)` applies
        # LayerNorm immediately after (see this arm's own test). It should land
        # on ~0.0 delta; if it does not, the harness itself is suspect.
        "scaled_scores_control": {"scale_scores": True},
        # Upstream uses raw scores @ V with no normalization at all.
        "softmax_attention": {"use_softmax": True},
        # Softmax WITH 1/sqrt(d) temperature -- scaling only becomes a real
        # lever in the presence of a softmax.
        "softmax_scaled": {"use_softmax": True, "scale_scores": True},
        # The full "make BDH's attention conventional" arm: softmax, proper
        # temperature, and a standard self-inclusive causal mask together.
        "standard_attention": {"use_softmax": True, "scale_scores": True, "mask_diagonal": 0},
    }
    del probe

    arms = {}
    for name, variant in variants.items():
        model = build_model()
        print(f"[{name}] starting on {device} ...", flush=True)
        result = train_variant(name, model, variant, args, stages, device)
        arms[name] = result
        print(f"[{name}] best_val={result['best_validation_loss']:.4f} "
              f"seconds={result['training_seconds']:.0f}", flush=True)
        del model

    baseline_loss = arms["baseline_upstream"]["best_validation_loss"]
    for arm in arms.values():
        arm["validation_loss_minus_baseline"] = arm["best_validation_loss"] - baseline_loss

    report = {
        "device": str(device),
        "dtype": "float32",
        "scaled_down_local_run": True,
        "not_comparable_to_cuda_reference_numbers": (
            "Absolute losses are NOT comparable to the 25M-token CUDA numbers "
            "(dense BDH 1.3848, matched Transformer 1.5141). Only the DIRECTION and "
            "rough magnitude of each primitive's effect is the real signal here."
        ),
        "baseline_is_bitwise_oracle": (
            "The baseline arm is proven bit-for-bit identical to BDH.forward by "
            "tests/reference/test_hz0h_bdh_primitive_ablations_torch.py"
        ),
        "curriculum_stages": stages,
        "target_tokens": args.target_tokens,
        "shape": {
            "n_embd": args.n_embd, "n_layer": args.n_layer, "n_head": args.n_head,
            "mlp_internal_dim_multiplier": args.mlp_internal_dim_multiplier,
            "batch_size": args.batch_size, "sequence_length": args.sequence_length,
        },
        "seed": args.seed,
        "arms": arms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({n: {
        "best_validation_loss": round(a["best_validation_loss"], 4),
        "vs_baseline": round(a["validation_loss_minus_baseline"], 4),
    } for n, a in arms.items()}, indent=2))


if __name__ == "__main__":
    main()
