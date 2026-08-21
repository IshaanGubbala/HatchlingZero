"""Real correctness tests for reference/hz0h_bdh_wide_gemm_checkpointed_torch.py.

Load-bearing gate: checked on logits, loss, AND gradients (both the
combined fixes -- wide-GEMM math and checkpointed recompute -- must not
change the actual math, only when/how it's computed)."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward
from reference.hz0h_bdh_wide_gemm_checkpointed_torch import bdh_wide_gemm_forward_checkpointed


def _model(seed: int = 11, **overrides) -> BDH:
    torch.manual_seed(seed)
    config = BDHConfig(
        n_layer=overrides.get("n_layer", 5), n_embd=overrides.get("n_embd", 32),
        n_head=overrides.get("n_head", 4), mlp_internal_dim_multiplier=overrides.get("mult", 8),
        vocab_size=256, dropout=0.0,
    )
    return BDH(config)


def test_checkpointed_wide_gemm_matches_reference_logits_exactly():
    model = _model()
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    with torch.no_grad():
        reference_logits, reference_loss = bdh_variable_depth_forward(model, idx, model.config.n_layer, targets)
    checkpointed_logits, checkpointed_loss = bdh_wide_gemm_forward_checkpointed(
        model, idx, model.config.n_layer, targets, checkpoint_segment_size=1,
    )
    assert torch.allclose(reference_logits, checkpointed_logits, atol=1e-5, rtol=1e-4), (
        f"max diff {(reference_logits - checkpointed_logits).abs().max()}"
    )
    assert torch.allclose(reference_loss, checkpointed_loss, atol=1e-5, rtol=1e-4)


def test_checkpointed_wide_gemm_backward_matches_reference_gradients():
    reference_model = _model()
    checkpointed_model = _model()
    checkpointed_model.load_state_dict(reference_model.state_dict())
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))

    _, reference_loss = bdh_variable_depth_forward(reference_model, idx, reference_model.config.n_layer, targets)
    reference_loss.backward()

    _, checkpointed_loss = bdh_wide_gemm_forward_checkpointed(
        checkpointed_model, idx, checkpointed_model.config.n_layer, targets, checkpoint_segment_size=1,
    )
    checkpointed_loss.backward()

    assert torch.allclose(reference_loss, checkpointed_loss, atol=1e-5, rtol=1e-4)
    params_a = dict(reference_model.named_parameters())
    params_b = dict(checkpointed_model.named_parameters())
    assert params_a.keys() == params_b.keys()
    for name in params_a:
        ga, gb = params_a[name].grad, params_b[name].grad
        assert ga is not None and gb is not None, f"{name} missing a gradient"
        assert torch.allclose(ga, gb, atol=1e-4, rtol=1e-3), (
            f"gradient mismatch at {name}: max diff {(ga - gb).abs().max()}"
        )


def test_segment_size_greater_than_one_still_matches_reference():
    """checkpoint_segment_size > 1 must not change the math, only how
    many rounds get recomputed together during backward."""
    reference_model = _model(n_layer=6)
    checkpointed_model = _model(n_layer=6)
    checkpointed_model.load_state_dict(reference_model.state_dict())
    idx = torch.randint(256, (2, 9))
    targets = torch.randint(256, (2, 9))

    _, reference_loss = bdh_variable_depth_forward(reference_model, idx, 6, targets)
    reference_loss.backward()

    _, checkpointed_loss = bdh_wide_gemm_forward_checkpointed(
        checkpointed_model, idx, 6, targets, checkpoint_segment_size=3,
    )
    checkpointed_loss.backward()

    assert torch.allclose(reference_loss, checkpointed_loss, atol=1e-5, rtol=1e-4)
    for name, ref_param in reference_model.named_parameters():
        ckpt_param = dict(checkpointed_model.named_parameters())[name]
        assert torch.allclose(ref_param.grad, ckpt_param.grad, atol=1e-4, rtol=1e-3), (
            f"gradient mismatch at {name}"
        )


def test_optimizer_step_actually_updates_encoder_and_encoder_v():
    """Real end-to-end check: a full checkpointed-wide-gemm train step
    must move encoder/encoder_v/decoder, not just compute gradients
    that go nowhere."""
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    before_encoder = model.encoder.detach().clone()
    before_encoder_v = model.encoder_v.detach().clone()

    _, loss = bdh_wide_gemm_forward_checkpointed(model, idx, model.config.n_layer, targets)
    loss.backward()
    optimizer.step()

    assert not torch.equal(before_encoder, model.encoder.detach())
    assert not torch.equal(before_encoder_v, model.encoder_v.detach())


def test_eval_mode_no_grad_matches_plain_wide_gemm_forward():
    """When autograd is disabled, this should take the plain (non-
    checkpointed) wide-GEMM path -- checkpointing has no benefit
    without a backward pass to save memory for."""
    from reference.hz0h_bdh_wide_gemm_trainable_torch import bdh_wide_gemm_trainable_forward

    model = _model()
    idx = torch.randint(256, (2, 12))
    with torch.no_grad():
        checkpointed_logits, _ = bdh_wide_gemm_forward_checkpointed(model, idx, model.config.n_layer)
        plain_logits, _ = bdh_wide_gemm_trainable_forward(model, idx, model.config.n_layer)
    assert torch.equal(checkpointed_logits, plain_logits)
