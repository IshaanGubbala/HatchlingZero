"""HZ-0D D10: evaluation tests. Checked against the ACTUAL frozen
checkpoint and REAL corpus text, matching this project's established
convention. Skips if either isn't present locally. Locks in the two
genuinely NEW measurements D10 needed beyond what D1-D9 already
verified: general-quality degradation on UNRELATED real text, and a
real-scale (dim=768/rank=16) confirmation of D10's own exit gate
("HZ-0D beats prompting, memory-only, and static-adapter baselines").
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from reference.hz0b_b8_latent_write import init_latent_write_controller
from reference.hz0d_d6_integration import d6_fast_weight_config, d6_forward_with_fast_weights
from reference.hz0d_d7_state_ordering import d7_process_sequence
from reference.hz0d_d8_curriculum import make_natural_schema_task
from reference.hz0d_fast_weights import FastWeightConfig, FastWeightState, init_fast_weights
from reference.hz0d_fair_baselines import (
    in_context_attention_baseline, knn_retrieval_baseline, longer_context_baseline,
    no_adaptation_baseline, static_random_adapter_baseline,
)
from reference.hz0d_isolated_simulator import held_out_generalization_loss
from reference.hz0d_update_mechanisms import delta_prediction_update
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences
from scripts.hz0c_c6_conditional_attention_eval import (
    conditional_forward_with_memory, fixed_matched_trigger, loss_and_ppl,
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists() or not Path(GENERAL_DATA_PATH).exists(),
    reason="frozen HZ-0A checkpoint / real corpus data not present locally (both gitignored)",
)

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


def test_full_pipeline_with_inactive_fast_weights_preserves_hz0b_and_hz0c_behavior_exactly():
    """Completion item 4 ("HZ-0B and HZ-0C behavior is preserved"),
    checked at the FULL composed-pipeline level, not just D6's
    memory-free comparison: D7's `d7_process_sequence` with inactive
    fast weights must be BIT-IDENTICAL to HZ-0C's own
    `conditional_forward_with_memory` (which already wires in real
    HZ-0B memory) -- on real checkpoint logits, a real trigger, and the
    real memory write gates, not approximately close."""
    model, _payload = load_frozen_model()
    config = d6_fast_weight_config()
    fast_state = init_fast_weights(config)
    latent_params = init_latent_write_controller(d_model=768, key_dim=64, value_dim=64, seed=0)
    tokens = mx.array([[1, 45, 982, 12, 7, 300, 44, 1023, 55, 66, 77, 88]])
    trigger = fixed_matched_trigger(1, tokens.shape[1], 0.15)

    d7_result = d7_process_sequence(model, tokens, trigger, latent_params, fast_state, config)
    hz0c_logits, hz0c_write_gates = conditional_forward_with_memory(model, tokens, trigger, latent_params)
    mx.eval(d7_result.logits, hz0c_logits)

    assert bool(mx.array_equal(d7_result.logits, hz0c_logits))
    assert bool(mx.array_equal(d7_result.write_gates, hz0c_write_gates))


def test_benign_active_adaptation_barely_affects_unrelated_real_text_quality():
    """General-quality degradation, the one D10 dimension not yet
    measured by D1-D9: fit fast weights to a real, benign task, then
    measure LM loss on DIFFERENT real corpus text the adaptation never
    saw. Degradation must be small (<5% relative) -- fast weights only
    activate at triggered positions (15% rate) across 6 of many layers,
    so unrelated text should barely notice."""
    model, _payload = load_frozen_model()
    config = d6_fast_weight_config()
    adapt_tokens = _load_real_tokens(0, 8)
    probe_tokens = _load_real_tokens(8, 8)
    probe_trigger = fixed_matched_trigger(probe_tokens.shape[0], probe_tokens.shape[1], 0.15)

    task = make_natural_schema_task(model, adapt_tokens, heads=model.heads, seed=0, rule_scale=0.05, k_train=256, k_held_out=64)
    adapted_single, _ = delta_prediction_update(task, init_fast_weights(SINGLE_LAYER_CONFIG), SINGLE_LAYER_CONFIG)
    full_state = _full_state_with_layer0(config, adapted_single)
    inactive_state = init_fast_weights(config)

    inactive_logits = d6_forward_with_fast_weights(model, probe_tokens, probe_trigger, inactive_state, config)
    active_logits = d6_forward_with_fast_weights(model, probe_tokens, probe_trigger, full_state, config)
    inactive_loss, _ = loss_and_ppl(inactive_logits, probe_tokens)
    active_loss, _ = loss_and_ppl(active_logits, probe_tokens)

    relative_delta = abs(active_loss - inactive_loss) / inactive_loss
    assert relative_delta < 0.05, f"expected <5% general-quality change from a benign adapted state: {relative_delta:.4f}"


def test_adversarial_clipped_state_does_not_amplify_general_quality_harm():
    """The clip bound's real payoff: even an ADVERSARIAL rule
    (rule_scale=1000), fit under D1's real production
    max_delta_norm=1.0, must not cause outsized damage to UNRELATED
    real text quality -- the safety bound contains the blast radius,
    not just the realized delta's own norm."""
    model, _payload = load_frozen_model()
    config = d6_fast_weight_config()  # real production max_delta_norm=1.0
    adapt_tokens = _load_real_tokens(0, 8)
    probe_tokens = _load_real_tokens(8, 8)
    probe_trigger = fixed_matched_trigger(probe_tokens.shape[0], probe_tokens.shape[1], 0.15)

    adversarial_config = FastWeightConfig(dim=768, rank=16, num_layers=1, max_delta_norm=config.max_delta_norm)
    adv_task = make_natural_schema_task(model, adapt_tokens, heads=model.heads, seed=99, rule_scale=1000.0, k_train=64, k_held_out=16)
    adv_single, _ = delta_prediction_update(adv_task, init_fast_weights(adversarial_config), adversarial_config)
    delta_norm = float(mx.sqrt(mx.sum((adv_single.a_fast[0] @ adv_single.b_fast[0]) ** 2)))
    assert delta_norm <= config.max_delta_norm + 1e-3

    adv_full_state = _full_state_with_layer0(config, adv_single)
    inactive_state = init_fast_weights(config)
    inactive_logits = d6_forward_with_fast_weights(model, probe_tokens, probe_trigger, inactive_state, config)
    adv_logits = d6_forward_with_fast_weights(model, probe_tokens, probe_trigger, adv_full_state, config)
    inactive_loss, _ = loss_and_ppl(inactive_logits, probe_tokens)
    adv_loss, _ = loss_and_ppl(adv_logits, probe_tokens)

    relative_delta = abs(adv_loss - inactive_loss) / inactive_loss
    assert relative_delta < 0.05, f"expected the clip bound to contain adversarial general-quality harm to <5%: {relative_delta:.4f}"


def test_selected_mechanism_beats_all_named_baseline_categories_at_real_scale():
    """D10's own exit gate, confirmed at real dim=768/rank=16 scale
    (D4 only checked the isolated dim=8 toy task): D3's selected
    delta-prediction (v4) mechanism must beat prompting (in-context
    attention, longer context), memory-only (k-NN retrieval), and
    static-adapter (a frozen random low-rank adapter) baselines, using
    the REAL frozen output-projection weight as every task's base."""
    model, _payload = load_frozen_model()
    tokens = _load_real_tokens(0, 8)

    def task_factory(seed):
        return make_natural_schema_task(model, tokens, heads=model.heads, seed=seed, rule_scale=0.05, k_train=128, k_held_out=64)

    baseline_losses = {"no_adaptation": [], "static_random_adapter": [], "in_context_attention": [], "longer_context": [], "knn_retrieval": []}
    delta_losses = []
    for seed in range(5):
        task = task_factory(seed)
        baseline_losses["no_adaptation"].append(no_adaptation_baseline(task, SINGLE_LAYER_CONFIG)[0])
        baseline_losses["static_random_adapter"].append(static_random_adapter_baseline(SINGLE_LAYER_CONFIG, task, seed=seed)[0])
        baseline_losses["in_context_attention"].append(in_context_attention_baseline(task, SINGLE_LAYER_CONFIG)[0])
        baseline_losses["longer_context"].append(longer_context_baseline(task, SINGLE_LAYER_CONFIG, extra_k=128, seed=1000 + seed)[0])
        baseline_losses["knn_retrieval"].append(knn_retrieval_baseline(task, SINGLE_LAYER_CONFIG, k=3)[0])
        dstate, _ = delta_prediction_update(task, init_fast_weights(SINGLE_LAYER_CONFIG), SINGLE_LAYER_CONFIG)
        delta_losses.append(held_out_generalization_loss(task, dstate))

    mean_delta = sum(delta_losses) / len(delta_losses)
    for name, values in baseline_losses.items():
        mean_baseline = sum(values) / len(values)
        assert mean_delta < mean_baseline * 0.5, f"delta prediction should clearly beat {name} at real scale: delta={mean_delta} {name}={mean_baseline}"


def test_gradient_descent_shows_real_instability_gd_lr_tuned_delta_prediction_does_not():
    """A real, disclosed finding: even at a learning rate re-tuned FOR
    this task type (real corpus-derived activations, unlike D6's
    synthetic Gaussian ones -- lr=3.0 there, lr=0.1 here), gradient
    descent still diverges on a real fraction of task instances
    (different random low-rank rules), while delta prediction (v4)
    stays uniformly stable. Not a strawman: same task family, same
    config, only the update mechanism differs. This is additional real
    evidence for D3's mechanism selection, found during D10 rather than
    assumed to not matter at real scale."""
    model, _payload = load_frozen_model()
    tokens = _load_real_tokens(0, 8)

    def task_factory(seed):
        return make_natural_schema_task(model, tokens, heads=model.heads, seed=seed, rule_scale=0.05, k_train=128, k_held_out=64)

    from reference.hz0d_update_mechanisms import gradient_descent_update

    delta_losses, gd_losses = [], []
    for seed in range(8):
        task = task_factory(seed)
        dstate, _ = delta_prediction_update(task, init_fast_weights(SINGLE_LAYER_CONFIG), SINGLE_LAYER_CONFIG)
        delta_losses.append(held_out_generalization_loss(task, dstate))
        gstate, _ = gradient_descent_update(task, init_fast_weights(SINGLE_LAYER_CONFIG), SINGLE_LAYER_CONFIG, steps=400, lr=0.1)
        gd_losses.append(held_out_generalization_loss(task, gstate))

    assert max(delta_losses) < 2.0, f"delta prediction should stay uniformly stable across seeds: {delta_losses}"
    diverged = sum(1 for loss in gd_losses if loss > 10.0)
    assert diverged >= 1, (
        f"expected at least one real GD divergence at this tuned lr, matching the disclosed finding "
        f"(if this now fails, gradient descent's real-scale stability may have genuinely improved -- verify before assuming a bug): {gd_losses}"
    )
