"""Real correctness tests for reference/hz0h_bdh_g_r_operator_diagnostic_torch.py:
does capturing g_r per round change BDH's actual math? Checked on
logits AND loss, not just shape."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_g_r_operator_diagnostic_torch import bdh_forward_with_g_r
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward


def _model(seed: int = 3, **overrides) -> BDH:
    torch.manual_seed(seed)
    config = BDHConfig(
        n_layer=overrides.get("n_layer", 6), n_embd=overrides.get("n_embd", 32),
        n_head=overrides.get("n_head", 4), mlp_internal_dim_multiplier=overrides.get("mult", 8),
        vocab_size=256, dropout=0.0,
    )
    return BDH(config)


def test_g_r_capture_matches_reference_logits_and_loss_exactly():
    model = _model()
    model.eval()
    idx = torch.randint(256, (2, 9))
    targets = torch.randint(256, (2, 9))
    with torch.no_grad():
        reference_logits, reference_loss = bdh_variable_depth_forward(model, idx, model.config.n_layer, targets)
        captured_logits, captured_loss, g_states = bdh_forward_with_g_r(model, idx, model.config.n_layer, targets)
    assert torch.allclose(reference_logits, captured_logits, atol=1e-6, rtol=1e-5)
    assert torch.allclose(reference_loss, captured_loss, atol=1e-6, rtol=1e-5)


def test_g_r_states_have_expected_shape_and_count():
    model = _model(n_layer=5, n_embd=32, n_head=4, mult=8)
    model.eval()
    idx = torch.randint(256, (3, 7))
    N = 32 * 8 // 4
    with torch.no_grad():
        _, _, g_states = bdh_forward_with_g_r(model, idx, model.config.n_layer)
    assert len(g_states) == model.config.n_layer
    for g in g_states:
        assert g.shape == (3, 4, 7, N)
        assert torch.isfinite(g).all()


def test_g_r_is_nonnegative_since_it_is_a_product_of_two_relus():
    """g_r = ReLU(.) * ReLU(.) -- must be elementwise >= 0. A real bug in
    the capture point (e.g. capturing after some other op) would likely
    break this invariant silently."""
    model = _model()
    model.eval()
    idx = torch.randint(256, (2, 9))
    with torch.no_grad():
        _, _, g_states = bdh_forward_with_g_r(model, idx, model.config.n_layer)
    for g in g_states:
        assert (g >= 0).all()
