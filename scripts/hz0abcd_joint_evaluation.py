"""Bounded joint HZ-0A/B/C/D evaluation on disjoint real-corpus sequences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from reference.hz0b_b8_latent_write import init_latent_write_controller
from reference.hz0d_d6_integration import d6_fast_weight_config
from reference.hz0d_d7_state_ordering import d7_process_sequence
from reference.hz0d_fast_weights import init_fast_weights
from reference.hz0d_d8_curriculum import make_natural_schema_task
from scripts.hz0b_b11_baseline_comparison import load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences
from scripts.hz0c_c6_conditional_attention_eval import fixed_matched_trigger, loss_and_ppl


def _tokens(start: int, count: int) -> mx.array:
    rows = load_real_sequences(GENERAL_DATA_PATH, start + count)[start:]
    width = min(len(row) for row in rows)
    return mx.array([row[:width] for row in rows], dtype=mx.int32)


def run_seed(model, seed: int, *, start: int, count: int) -> dict:
    adapt = _tokens(start, count)
    evaluate = _tokens(start + count, count)
    trigger = fixed_matched_trigger(*evaluate.shape, rate=0.15)
    latent = init_latent_write_controller(model.dim, 32, 32, seed=seed, write_gate_bias_init=-3.0)
    config = d6_fast_weight_config()
    inactive = init_fast_weights(config)
    task = make_natural_schema_task(model, adapt, heads=model.heads, seed=seed, rule_scale=0.05, k_train=128, k_held_out=64)

    baseline = d7_process_sequence(model, evaluate, trigger, latent, inactive, config)
    adapted = d7_process_sequence(
        model, evaluate, trigger, latent, inactive, config,
        fast_update_layer_index=0, fast_update_task=task,
    )
    base_loss, _ = loss_and_ppl(baseline.logits, evaluate)
    # D7 deliberately applies the update after producing this call's logits;
    # evaluate the returned state on the next causal call to avoid feedback.
    applied = d7_process_sequence(model, evaluate, trigger, latent, adapted.fast_state, config)
    adapted_loss, _ = loss_and_ppl(applied.logits, evaluate)
    mx.eval(base_loss, adapted_loss, baseline.memory_state.keys, applied.memory_state.keys, adapted.fast_state.a_fast)
    update_norm = float(mx.sqrt(mx.sum(adapted.fast_state.a_fast ** 2) + mx.sum(adapted.fast_state.b_fast ** 2)))
    row = {
        "seed": seed,
        "baseline_loss": float(base_loss),
        "adapted_loss": float(adapted_loss),
        "loss_delta": float(base_loss - adapted_loss),
        "memory_state_delta": float(mx.max(mx.abs(baseline.memory_state.keys - applied.memory_state.keys))),
        "fast_update_norm": update_norm,
        "fast_weight_updated": adapted.fast_weight_updated,
        "finite": bool(mx.all(mx.isfinite(baseline.logits)) and mx.all(mx.isfinite(adapted.logits))),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[555, 556, 557])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    model, _ = load_frozen_model()
    rows = [run_seed(model, seed, start=args.start, count=args.count) for seed in args.seeds]
    deltas = [row["loss_delta"] for row in rows]
    report = {
        "protocol": {"seeds": args.seeds, "adapt_sequences": args.count, "eval_sequences": args.count, "trigger_rate": 0.15, "sequential": True},
        "components": {"hz0a": "frozen 301M checkpoint", "hz0b": "latent write/read memory", "hz0c": "fixed 15% surprise-trigger schedule", "hz0d": "bounded delta-prediction update"},
        "seeds": rows,
        "summary": {"mean_loss_delta": sum(deltas) / len(deltas), "wins": sum(delta > 0 for delta in deltas), "finite": all(row["finite"] for row in rows), "all_updates_applied": all(row["fast_weight_updated"] for row in rows)},
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
