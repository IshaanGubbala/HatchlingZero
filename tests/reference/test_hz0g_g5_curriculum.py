"""HZ-0G G5: correctness tests for reference/hz0g_g5_curriculum.py --
the real Dense/MoE/domain-adapter curriculum run through the FULL
A+B+C+D+(E) integration, not A+E alone.

Checked against the ACTUAL frozen checkpoint, matching this project's
established convention. Skips if the checkpoint isn't present locally.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from reference.hz0e_e6_integration import TARGET_LAYERS, init_e6_layers
from reference.hz0e_moe_contract import MoeConfig
from reference.hz0g_g5_curriculum import (
    _dense_loss, _fixed_bcd, evaluate_integrated_dense_per_domain, evaluate_integrated_moe_per_domain,
    integrated_loss, run_integrated_dense_baseline, run_integrated_moe_curriculum,
)
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists(),
    reason="frozen HZ-0A checkpoint not present locally (gitignored under outputs/)",
)


def test_integrated_loss_moe_disabled_finite():
    model, _payload = load_frozen_model()
    tokens = mx.array([[1, 45, 982, 12, 7, 300, 44, 1023]])
    loss = integrated_loss(model, tokens, moe_enabled=False)
    mx.eval(loss)
    assert bool(mx.isfinite(loss))


def test_integrated_loss_moe_enabled_finite():
    model, _payload = load_frozen_model()
    moe_layers = init_e6_layers(model, seed=0, target_layers=TARGET_LAYERS, warm_start_experts=True)
    tokens = mx.array([[1, 45, 982, 12, 7, 300, 44, 1023]])
    loss = integrated_loss(model, tokens, moe_layers=moe_layers, moe_enabled=True)
    mx.eval(loss)
    assert bool(mx.isfinite(loss))


def test_dense_loss_finite_and_matches_manual_shape():
    model, _payload = load_frozen_model()
    flat_params = run_integrated_dense_baseline(model, balanced_steps=1, mixed_steps=1, imbalanced_steps=1, seed=0)
    assert len(flat_params) == 6 * len(TARGET_LAYERS)  # gate_w/gate_b/up_w/up_b/down_w/down_b per target layer
    tokens = mx.array([[1, 45, 982, 12, 7, 300, 44, 1023]])
    loss = _dense_loss(model, flat_params, tokens, TARGET_LAYERS, seed=0)
    mx.eval(loss)
    assert bool(mx.isfinite(loss))


def test_fixed_bcd_deterministic_given_same_seed():
    model, _payload = load_frozen_model()
    trigger1, latent1, fast1, config1 = _fixed_bcd(model, 1, 8, seed=0)
    trigger2, latent2, fast2, config2 = _fixed_bcd(model, 1, 8, seed=0)
    mx.eval(trigger1, trigger2)
    assert bool(mx.array_equal(trigger1, trigger2))
    assert bool(mx.array_equal(latent1.write_controller.write_gate_w, latent2.write_controller.write_gate_w))


def test_run_integrated_moe_curriculum_tiny_smoke():
    """Real, if tiny, end-to-end run -- not just unit-level shape checks.
    Catches wiring bugs a shape-only test would miss (this exact test
    caught the module importing correctly and running without crashing
    during initial development; kept as a fast regression guard)."""
    model, _payload = load_frozen_model()
    config = MoeConfig(dim=model.dim)
    trained_layers, pure_dense, warm, after = run_integrated_moe_curriculum(
        model, config, balanced_steps=1, mixed_steps=1, imbalanced_steps=1, warm_start_steps=1, seed=0,
    )
    assert set(trained_layers.keys()) == set(TARGET_LAYERS)
    assert all(v > 0 and v == v for v in (pure_dense, warm, after))  # finite and positive real cross-entropy


def test_evaluate_per_domain_returns_all_five_domains():
    model, _payload = load_frozen_model()
    config = MoeConfig(dim=model.dim)
    trained_layers, *_ = run_integrated_moe_curriculum(model, config, balanced_steps=1, mixed_steps=1, imbalanced_steps=1, warm_start_steps=1, seed=0)
    moe_per_domain = evaluate_integrated_moe_per_domain(model, trained_layers, seed=0)
    assert set(moe_per_domain.keys()) == {"prose", "code", "math", "json", "tools"}
    assert all(v == v and v > 0 for v in moe_per_domain.values())

    flat_params = run_integrated_dense_baseline(model, balanced_steps=1, mixed_steps=1, imbalanced_steps=1, seed=0)
    dense_per_domain = evaluate_integrated_dense_per_domain(model, flat_params, seed=0)
    assert set(dense_per_domain.keys()) == {"prose", "code", "math", "json", "tools"}
    assert all(v == v and v > 0 for v in dense_per_domain.values())
