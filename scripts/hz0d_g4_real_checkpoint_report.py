"""HZ-0G G4: real-number companion to tests/reference/test_hz0d_d10_evaluation.py.

That test file only asserts pass/fail; this script runs the exact same
logic (same functions, same seeds, same configs) and prints the actual
measured values, matching the reporting rigor used for G2/G3. Not new
logic -- a visibility layer over already-tested code.
"""
from __future__ import annotations

import mlx.core as mx

from reference.hz0b_b8_latent_write import init_latent_write_controller  # noqa: F401 (parity import with the test file)
from reference.hz0d_d6_integration import d6_fast_weight_config, d6_forward_with_fast_weights
from reference.hz0d_d8_curriculum import make_natural_schema_task
from reference.hz0d_fast_weights import FastWeightConfig, FastWeightState, init_fast_weights
from reference.hz0d_fair_baselines import (
    in_context_attention_baseline, knn_retrieval_baseline, longer_context_baseline,
    no_adaptation_baseline, static_random_adapter_baseline,
)
from reference.hz0d_isolated_simulator import held_out_generalization_loss
from reference.hz0d_update_mechanisms import delta_prediction_update, gradient_descent_update
from scripts.hz0b_b11_baseline_comparison import load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences
from scripts.hz0c_c6_conditional_attention_eval import fixed_matched_trigger, loss_and_ppl

SINGLE_LAYER_CONFIG = FastWeightConfig(dim=768, rank=16, num_layers=1, max_delta_norm=10.0)


def _load_real_tokens(start: int, count: int) -> mx.array:
    sequences = load_real_sequences(GENERAL_DATA_PATH, start + count)[start:]
    min_len = min(len(s) for s in sequences)
    return mx.array([s[:min_len] for s in sequences])


def _full_state_with_layer0(config, single_layer_state):
    return FastWeightState(
        a_fast=mx.zeros((config.num_layers, config.dim, config.rank)).at[0].add(single_layer_state.a_fast[0]),
        b_fast=mx.zeros((config.num_layers, config.rank, config.dim)).at[0].add(single_layer_state.b_fast[0]),
        update_count=mx.array(1, dtype=mx.int32),
    )


def main() -> None:
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")
    config = d6_fast_weight_config()
    adapt_tokens = _load_real_tokens(0, 8)
    probe_tokens = _load_real_tokens(8, 8)
    probe_trigger = fixed_matched_trigger(probe_tokens.shape[0], probe_tokens.shape[1], 0.15)

    print("\n1. Benign adaptation -- unrelated-text degradation")
    task = make_natural_schema_task(model, adapt_tokens, heads=model.heads, seed=0, rule_scale=0.05, k_train=256, k_held_out=64)
    adapted_single, _ = delta_prediction_update(task, init_fast_weights(SINGLE_LAYER_CONFIG), SINGLE_LAYER_CONFIG)
    full_state = _full_state_with_layer0(config, adapted_single)
    inactive_state = init_fast_weights(config)
    inactive_logits = d6_forward_with_fast_weights(model, probe_tokens, probe_trigger, inactive_state, config)
    active_logits = d6_forward_with_fast_weights(model, probe_tokens, probe_trigger, full_state, config)
    inactive_loss, _ = loss_and_ppl(inactive_logits, probe_tokens)
    active_loss, _ = loss_and_ppl(active_logits, probe_tokens)
    relative_delta = abs(active_loss - inactive_loss) / inactive_loss
    print(f"  inactive_loss={inactive_loss:.6f} active_loss={active_loss:.6f} relative_delta={relative_delta:.4f} (gate: <0.05)")

    print("\n2. Adversarial clipped state -- unrelated-text harm containment")
    adversarial_config = FastWeightConfig(dim=768, rank=16, num_layers=1, max_delta_norm=config.max_delta_norm)
    adv_task = make_natural_schema_task(model, adapt_tokens, heads=model.heads, seed=99, rule_scale=1000.0, k_train=64, k_held_out=16)
    adv_single, _ = delta_prediction_update(adv_task, init_fast_weights(adversarial_config), adversarial_config)
    delta_norm = float(mx.sqrt(mx.sum((adv_single.a_fast[0] @ adv_single.b_fast[0]) ** 2)))
    adv_full_state = _full_state_with_layer0(config, adv_single)
    adv_logits = d6_forward_with_fast_weights(model, probe_tokens, probe_trigger, adv_full_state, config)
    adv_loss, _ = loss_and_ppl(adv_logits, probe_tokens)
    adv_relative_delta = abs(adv_loss - inactive_loss) / inactive_loss
    print(f"  delta_norm={delta_norm:.4f} (bound={config.max_delta_norm}) adv_loss={adv_loss:.6f} relative_delta={adv_relative_delta:.4f} (gate: <0.05)")

    print("\n3. Delta prediction vs. named baselines (5 seeds, real dim=768/rank=16 scale)")
    tokens = _load_real_tokens(0, 8)

    def task_factory(seed):
        return make_natural_schema_task(model, tokens, heads=model.heads, seed=seed, rule_scale=0.05, k_train=128, k_held_out=64)

    baseline_losses = {"no_adaptation": [], "static_random_adapter": [], "in_context_attention": [], "longer_context": [], "knn_retrieval": []}
    delta_losses = []
    for seed in range(5):
        t = task_factory(seed)
        baseline_losses["no_adaptation"].append(no_adaptation_baseline(t, SINGLE_LAYER_CONFIG)[0])
        baseline_losses["static_random_adapter"].append(static_random_adapter_baseline(SINGLE_LAYER_CONFIG, t, seed=seed)[0])
        baseline_losses["in_context_attention"].append(in_context_attention_baseline(t, SINGLE_LAYER_CONFIG)[0])
        baseline_losses["longer_context"].append(longer_context_baseline(t, SINGLE_LAYER_CONFIG, extra_k=128, seed=1000 + seed)[0])
        baseline_losses["knn_retrieval"].append(knn_retrieval_baseline(t, SINGLE_LAYER_CONFIG, k=3)[0])
        dstate, _ = delta_prediction_update(t, init_fast_weights(SINGLE_LAYER_CONFIG), SINGLE_LAYER_CONFIG)
        delta_losses.append(held_out_generalization_loss(t, dstate))
    mean_delta = sum(delta_losses) / len(delta_losses)
    print(f"  delta_prediction mean_loss={mean_delta:.6f}")
    for name, values in baseline_losses.items():
        mean_baseline = sum(values) / len(values)
        beats = "YES" if mean_delta < mean_baseline * 0.5 else "no"
        print(f"  vs {name:24s} mean_loss={mean_baseline:.6f}  beats-by-2x={beats}")

    print("\n4. Gradient-descent instability vs. delta prediction (8 seeds)")
    delta_losses2, gd_losses = [], []
    for seed in range(8):
        t = task_factory(seed)
        dstate, _ = delta_prediction_update(t, init_fast_weights(SINGLE_LAYER_CONFIG), SINGLE_LAYER_CONFIG)
        delta_losses2.append(held_out_generalization_loss(t, dstate))
        gstate, _ = gradient_descent_update(t, init_fast_weights(SINGLE_LAYER_CONFIG), SINGLE_LAYER_CONFIG, steps=400, lr=0.1)
        gd_losses.append(held_out_generalization_loss(t, gstate))
    diverged = sum(1 for loss in gd_losses if loss > 10.0)
    print(f"  delta_prediction losses: {[round(x, 4) for x in delta_losses2]} (max={max(delta_losses2):.4f}, gate: <2.0)")
    print(f"  gradient_descent losses: {[round(x, 4) for x in gd_losses]} (diverged>10.0: {diverged}/8)")


if __name__ == "__main__":
    main()
