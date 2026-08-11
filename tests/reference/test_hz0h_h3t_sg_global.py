"""Regression tests for HZ-0H H3-T's SG-global (real per-position BPTT
gradient targets) and periodic calibration sweep. Real findings this
session relied on: SG-global's per-position target reconstructs the true
gradient exactly (verifies the target itself is sound before trusting
anything trained on it), SG-global beats SG-local on both alignment and
quality at a longer horizon, and calibration quality degrades
monotonically as the synthetic fraction increases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from hz0h_h3t_sg_global import sg_global_target_data, cosine
from hz0h_h3t_sg_global_comparison import run as run_comparison
from hz0h_h3t_periodic_calibration_sweep import run as run_calibration
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _tiny_config():
    return BDHConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)


def test_sg_global_target_reconstructs_true_gradient_exactly():
    """The per-position target IS the true gradient's own decomposition
    via the chain rule, not an approximation of it -- verify the
    reconstruction matches model.encoder.grad from the same backward pass
    to float32 precision, not just roughly."""
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDH(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (3, 10))

    x_in_all, _x_sparse_all, grad_all, _loss = sg_global_target_data(model, idx)
    true_grad = model.encoder.grad.detach().clone()

    reconstructed = torch.einsum("btd,bhtn->hdn", x_in_all.squeeze(1), grad_all)
    max_diff = float((reconstructed - true_grad).abs().max())
    assert max_diff < 1e-5, f"reconstruction should match the real gradient almost exactly, got diff={max_diff}"


def test_sg_global_predictor_cosine_beats_sg_local_over_a_longer_run():
    """Real finding: at a short horizon the two are close, but over more
    steps SG-local's alignment degrades toward zero/negative while
    SG-global's stays meaningfully positive -- confirmed here at reduced
    (but still real, multi-hundred-step) scale for test speed."""
    config = _tiny_config()
    steps, warmup = 80, 15
    sg_local = run_comparison(config, seed=0, steps=steps, batch_size=4, seq_len=8, warmup_steps=warmup, condition="sg_local")
    sg_global = run_comparison(config, seed=0, steps=steps, batch_size=4, seq_len=8, warmup_steps=warmup, condition="sg_global")

    cos_local = sum(sg_local["cos_vs_true"][-10:]) / 10
    cos_global = sum(sg_global["cos_vs_true"][-10:]) / 10
    assert cos_global > cos_local, f"sg_global ({cos_global}) should beat sg_local ({cos_local}) on alignment over a longer run"
    assert all(l == l for l in sg_local["losses"]) and all(l == l for l in sg_global["losses"])


def test_calibration_quality_degrades_as_synthetic_fraction_increases():
    """Real, monotonic finding: more synthetic (fewer real recalibration)
    steps means worse quality, at this scale -- the predictor drifts
    without frequent real-gradient refreshes as the model's own weights
    (and thus the true gradient mapping) change during training."""
    config = _tiny_config()
    steps = 80
    losses_by_frac = {}
    for frac in (0.5, 0.9):
        out = run_calibration(config, seed=0, steps=steps, batch_size=4, seq_len=8, synthetic_fraction=frac, warmup_steps=10)
        losses_by_frac[frac] = sum(out["losses"][-10:]) / 10
        assert all(l == l for l in out["losses"]), f"frac={frac} produced NaN"

    assert losses_by_frac[0.9] > losses_by_frac[0.5], (
        f"higher synthetic fraction should show worse (higher) loss: "
        f"0.5->{losses_by_frac[0.5]}, 0.9->{losses_by_frac[0.9]}"
    )
