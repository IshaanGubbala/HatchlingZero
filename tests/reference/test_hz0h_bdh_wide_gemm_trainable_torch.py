"""Real correctness tests for reference/hz0h_bdh_wide_gemm_trainable_torch.py.

Load-bearing gate: checked on BOTH logits AND gradients. Swapping in the
wide-GEMM/batched-GEMM layouts must not change the actual math -- a bug
here could produce correct-looking forward logits while silently
corrupting gradients through the new layout."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward
from reference.hz0h_bdh_wide_gemm_trainable_torch import bdh_wide_gemm_trainable_forward


def _model(seed: int = 7, **overrides) -> BDH:
    torch.manual_seed(seed)
    config = BDHConfig(
        n_layer=overrides.get("n_layer", 5), n_embd=overrides.get("n_embd", 32),
        n_head=overrides.get("n_head", 4), mlp_internal_dim_multiplier=overrides.get("mult", 8),
        vocab_size=256, dropout=0.0,
    )
    return BDH(config)


def test_wide_gemm_trainable_forward_matches_reference_logits_exactly():
    model = _model()
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    with torch.no_grad():
        reference_logits, reference_loss = bdh_variable_depth_forward(model, idx, model.config.n_layer, targets)
        wide_logits, wide_loss = bdh_wide_gemm_trainable_forward(model, idx, model.config.n_layer, targets)
    assert torch.allclose(reference_logits, wide_logits, atol=1e-5, rtol=1e-4), (
        f"max diff {(reference_logits - wide_logits).abs().max()}"
    )
    assert torch.allclose(reference_loss, wide_loss, atol=1e-5, rtol=1e-4)


def test_wide_gemm_trainable_backward_matches_reference_gradients():
    reference_model = _model()
    wide_model = _model()
    wide_model.load_state_dict(reference_model.state_dict())
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))

    _, reference_loss = bdh_variable_depth_forward(reference_model, idx, reference_model.config.n_layer, targets)
    reference_loss.backward()

    _, wide_loss = bdh_wide_gemm_trainable_forward(wide_model, idx, wide_model.config.n_layer, targets)
    wide_loss.backward()

    assert torch.allclose(reference_loss, wide_loss, atol=1e-5, rtol=1e-4)
    params_a = dict(reference_model.named_parameters())
    params_b = dict(wide_model.named_parameters())
    assert params_a.keys() == params_b.keys()
    for name in params_a:
        ga, gb = params_a[name].grad, params_b[name].grad
        assert ga is not None and gb is not None, f"{name} missing a gradient"
        assert torch.allclose(ga, gb, atol=1e-4, rtol=1e-3), (
            f"gradient mismatch at {name}: max diff {(ga - gb).abs().max()}"
        )


def test_optimizer_step_actually_updates_encoder_and_encoder_v():
    """Real end-to-end check: a full train step through this forward
    must move encoder/encoder_v/decoder, not just compute gradients that
    go nowhere (e.g. if the wide-GEMM path accidentally created a
    disconnected copy of the parameter instead of a true view)."""
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    before_encoder = model.encoder.detach().clone()
    before_encoder_v = model.encoder_v.detach().clone()

    _, loss = bdh_wide_gemm_trainable_forward(model, idx, model.config.n_layer, targets)
    loss.backward()
    optimizer.step()

    assert not torch.equal(before_encoder, model.encoder.detach())
    assert not torch.equal(before_encoder_v, model.encoder_v.detach())
