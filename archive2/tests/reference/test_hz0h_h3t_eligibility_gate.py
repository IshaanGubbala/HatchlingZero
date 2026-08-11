"""Regression tests for HZ-0H H3-T Stage 1's two eligibility-vs-true-gradient
diagnostics (scripts/hz0h_h3t_eligibility_gate.py,
scripts/hz0h_h3t_eligibility_gate_v2.py). These are real findings this
session relied on to decide whether to pursue a BDH-native training-law
search further -- pin them down so they don't silently drift or break.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from hz0h_h3t_eligibility_gate import compute_eligibility_trace, compute_true_gradient, cosine
from hz0h_h3t_eligibility_gate_v2 import compute_local_signal_pseudo_gradient
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _tiny_model():
    torch.manual_seed(0)
    config = BDHConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)
    model = BDH(config)
    model.eval()
    return model, config


def test_raw_hebbian_trace_matches_true_gradient_shape_and_is_near_orthogonal():
    """Real finding: raw Hebbian eligibility (pre * post, no learning signal)
    has essentially zero correlation with the true BPTT gradient -- the
    naive/cheapest hypothesis is falsified. Pin the near-zero result down
    with a loose bound (real magnitude varies with seed/scale; the point is
    it stays far from any real alignment, not that it's exactly 0)."""
    model, config = _tiny_model()
    idx = torch.randint(0, config.vocab_size, (3, 12))

    trace = compute_eligibility_trace(model, idx)
    grad, loss = compute_true_gradient(model, idx)

    assert trace.shape == grad.shape == (config.n_head, config.n_embd, config.mlp_internal_dim_multiplier * config.n_embd // config.n_head)
    assert torch.isfinite(trace).all() and torch.isfinite(grad).all()
    assert torch.isfinite(torch.tensor(loss))

    c = cosine(trace, grad)
    assert abs(c) < 0.3, f"raw Hebbian trace should be near-orthogonal to the true gradient, got cos={c}"


def test_local_signal_pseudo_gradient_shows_real_positive_alignment():
    """Real finding: a depth-truncated LOCAL learning signal (each layer's
    own stop-gradient readout to lm_head/loss, no information from later
    layers) recovers substantial, consistent alignment with the true
    full-depth gradient -- cos ~0.5 at the scale this was first measured.
    Loose bound since exact value depends on seed/scale, but must be
    clearly, substantially positive, not just "not zero"."""
    model, config = _tiny_model()
    idx = torch.randint(0, config.vocab_size, (3, 12))
    targets = idx

    pseudo_grad = compute_local_signal_pseudo_gradient(model, idx, targets)
    from hz0h_h3t_eligibility_gate_v2 import compute_true_gradient as compute_true_gradient_v2
    true_grad, loss = compute_true_gradient_v2(model, idx, targets)

    assert pseudo_grad.shape == true_grad.shape
    assert torch.isfinite(pseudo_grad).all() and torch.isfinite(true_grad).all()
    assert torch.isfinite(torch.tensor(loss))

    c = cosine(pseudo_grad, true_grad)
    assert c > 0.2, f"local-signal pseudo-gradient should show real positive alignment with the true gradient, got cos={c}"


def test_local_signal_beats_raw_hebbian_on_the_same_model():
    """The core comparative finding: adding a genuinely local (depth-
    truncated) learning signal is a real, substantial improvement over raw
    Hebbian eligibility alone, on the identical model/input -- not just
    two independently-plausible numbers."""
    model, config = _tiny_model()
    idx = torch.randint(0, config.vocab_size, (3, 12))

    trace = compute_eligibility_trace(model, idx)
    grad, _ = compute_true_gradient(model, idx)
    raw_cos = cosine(trace, grad)

    pseudo_grad = compute_local_signal_pseudo_gradient(model, idx, idx)
    from hz0h_h3t_eligibility_gate_v2 import compute_true_gradient as compute_true_gradient_v2
    true_grad, _ = compute_true_gradient_v2(model, idx, idx)
    local_cos = cosine(pseudo_grad, true_grad)

    assert local_cos > raw_cos + 0.2, f"local signal ({local_cos}) should substantially beat raw Hebbian ({raw_cos})"
