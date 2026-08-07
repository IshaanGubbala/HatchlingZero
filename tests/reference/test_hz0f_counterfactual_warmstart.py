"""HZ-0F: counterfactual-utility router warm-start/refinement
(`reference/hz0f_counterfactual_warmstart.py`).

See `docs/restart/hz0f_counterfactual_router_refinement_results.md` for
the full writeup. Locks in two real findings: (1) counterfactual labels
are degenerate (collapse to one expert) when computed on the identical-
broadcast expert initialization `init_e6_layers` produces -- refinement
only makes sense once experts have differentiated through real training;
(2) applied as a post-curriculum refinement, it improves per-domain win
count in most seeds but CONSISTENTLY worsens general/OOD quality in
every seed -- the same specialization-costs-generality signature this
investigation keeps finding, now shown to apply to router sharpening
specifically, not just architecture/fallback choices.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from reference.hz0e_e6_integration import init_e6_layers
from reference.hz0e_e8_curriculum import (
    LAYER, TRAIN_DOMAIN_DATA_PATHS, evaluate_dense_per_domain, evaluate_moe_per_domain,
    load_domain_batches, make_warm_dense_loss_fn, per_domain_mean_loss, run_curriculum, run_warm_dense_baseline,
)
from reference.hz0e_e3_routing_objectives import lm_forward_with_moe, params_to_dict
from reference.hz0e_moe_contract import MoeConfig
from reference.hz0f_counterfactual_warmstart import compute_counterfactual_labels, counterfactual_warm_start
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists() or not Path(GENERAL_DATA_PATH).exists(),
    reason="real HZ-0A checkpoint/corpus not present locally",
)


def test_counterfactual_labels_are_degenerate_on_identical_broadcast_experts():
    """Real precondition check: at init_e6_layers' own starting point,
    all experts are bit-identical (broadcast-initialized from the same
    dense-FFN slice), so counterfactual labels collapse to one expert --
    not a bug, a real property of the initialization that motivates
    applying refinement AFTER training, not as a from-scratch warm start."""
    model, _ = load_frozen_model()
    config = MoeConfig(dim=model.dim)
    train_domains = load_domain_batches(TRAIN_DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=0)
    start = init_e6_layers(model, seed=0)[LAYER]
    assert bool(mx.array_equal(start.expert_gate_w[0], start.expert_gate_w[1])), "experts should start identical"

    labels = compute_counterfactual_labels(model, start, config, LAYER, train_domains["prose"])
    mx.eval(labels)
    unique_labels = set(labels.reshape(-1).tolist())
    assert len(unique_labels) == 1, f"expected degenerate (single-expert) labels on identical experts, got {unique_labels}"


def test_counterfactual_refinement_after_training_improves_domain_win_but_costs_ood():
    model, _ = load_frozen_model()
    config = MoeConfig(dim=model.dim)
    train_domains = load_domain_batches(TRAIN_DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=0)
    general_val = [mx.array([s[:64]]) for s in load_real_sequences(GENERAL_DATA_PATH, 10)]

    ood_gap_deltas = []
    print("\nHZ-0F counterfactual router refinement (real checkpoint, real curriculum):")
    for seed in (0, 1, 2):
        baseline, _report = run_curriculum(model, config, balanced_steps=15, mixed_steps=15, imbalanced_steps=15, seed=seed, warm_start_steps=20)
        assert not bool(mx.array_equal(baseline.expert_gate_w[0], baseline.expert_gate_w[1])), "experts should have differentiated after real training"

        refined = counterfactual_warm_start(model, train_domains, config, layer_index=LAYER, steps=20, start_params=baseline)
        dense_trained, _before, _after = run_warm_dense_baseline(model, balanced_steps=15, mixed_steps=15, imbalanced_steps=15, seed=seed)
        dense_pd = evaluate_dense_per_domain(model, dense_trained, layer_index=LAYER)
        dense_loss_fn = make_warm_dense_loss_fn(model, LAYER)
        dense_general = sum(float(dense_loss_fn(dense_trained, tb)) for tb in general_val) / len(general_val)

        gaps = {}
        for name, params in [("baseline", baseline), ("refined", refined)]:
            moe_pd = evaluate_moe_per_domain(model, params, config, layer_index=LAYER)
            wins = sum(1 for k in moe_pd if moe_pd[k] < dense_pd[k])
            pdict = params_to_dict(params)
            moe_general = sum(float(lm_forward_with_moe(pdict, model, tb, config, LAYER)[0]) for tb in general_val) / len(general_val)
            gap = moe_general - dense_general
            gaps[name] = gap
            print(f"  seed={seed} {name}: domain_win={wins}/5 per_domain_mean={per_domain_mean_loss(moe_pd):.4f} ood_gap={gap:.4f}")
            assert wins >= 2, f"domain win count regressed unexpectedly: {wins}/5"

        ood_gap_deltas.append(gaps["refined"] - gaps["baseline"])

    assert all(delta > 0.0 for delta in ood_gap_deltas), (
        f"expected the real, reproducible finding to hold: refinement consistently worsens the OOD gap "
        f"(delta > 0 in every seed): {ood_gap_deltas}"
    )
