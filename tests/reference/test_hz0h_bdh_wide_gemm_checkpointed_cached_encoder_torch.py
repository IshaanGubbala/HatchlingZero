"""Real correctness tests for
reference/hz0h_bdh_wide_gemm_checkpointed_cached_encoder_torch.py.

Load-bearing gate: checked on logits, loss, gradients, AND a direct
bit-exact comparison against the existing (per-round-repacking)
checkpointed forward -- caching the encoder's wide view once per step must
not change the math at all, only how many times it gets rebuilt."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward
from reference.hz0h_bdh_wide_gemm_checkpointed_torch import bdh_wide_gemm_forward_checkpointed
from reference.hz0h_bdh_wide_gemm_checkpointed_cached_encoder_torch import (
    bdh_wide_gemm_forward_checkpointed_cached_encoder,
)


def _model(seed: int = 11, **overrides) -> BDH:
    torch.manual_seed(seed)
    config = BDHConfig(
        n_layer=overrides.get("n_layer", 5), n_embd=overrides.get("n_embd", 32),
        n_head=overrides.get("n_head", 4), mlp_internal_dim_multiplier=overrides.get("mult", 8),
        vocab_size=256, dropout=0.0,
    )
    return BDH(config)


def test_matches_reference_logits_exactly():
    model = _model()
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    with torch.no_grad():
        reference_logits, reference_loss = bdh_variable_depth_forward(model, idx, model.config.n_layer, targets)
    cached_logits, cached_loss = bdh_wide_gemm_forward_checkpointed_cached_encoder(
        model, idx, model.config.n_layer, targets, checkpoint_segment_size=1,
    )
    assert torch.allclose(reference_logits, cached_logits, atol=1e-5, rtol=1e-4)
    assert torch.allclose(reference_loss, cached_loss, atol=1e-5, rtol=1e-4)


def test_backward_matches_reference_gradients():
    reference_model = _model()
    cached_model = _model()
    cached_model.load_state_dict(reference_model.state_dict())
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))

    _, reference_loss = bdh_variable_depth_forward(reference_model, idx, reference_model.config.n_layer, targets)
    reference_loss.backward()

    _, cached_loss = bdh_wide_gemm_forward_checkpointed_cached_encoder(
        cached_model, idx, cached_model.config.n_layer, targets, checkpoint_segment_size=1,
    )
    cached_loss.backward()

    assert torch.allclose(reference_loss, cached_loss, atol=1e-5, rtol=1e-4)
    params_a = dict(reference_model.named_parameters())
    params_b = dict(cached_model.named_parameters())
    assert params_a.keys() == params_b.keys()
    for name in params_a:
        ga, gb = params_a[name].grad, params_b[name].grad
        assert ga is not None and gb is not None, f"{name} missing a gradient"
        assert torch.allclose(ga, gb, atol=1e-4, rtol=1e-3), (
            f"gradient mismatch at {name}: max diff {(ga - gb).abs().max()}"
        )


def test_matches_existing_checkpointed_forward_bit_exact_logits_and_gradients():
    """The real point of this file: caching the encoder wide view once per
    step (instead of once per round-recompute) must be bit-identical to
    the existing per-round-repacking checkpointed forward, not just close
    to the oracle -- this isolates that only the repacking frequency
    changed, nothing about the computation itself."""
    existing_model = _model(n_layer=6)
    cached_model = _model(n_layer=6)
    cached_model.load_state_dict(existing_model.state_dict())
    idx = torch.randint(256, (2, 9))
    targets = torch.randint(256, (2, 9))

    existing_logits, existing_loss = bdh_wide_gemm_forward_checkpointed(
        existing_model, idx, 6, targets, checkpoint_segment_size=1,
    )
    existing_loss.backward()

    cached_logits, cached_loss = bdh_wide_gemm_forward_checkpointed_cached_encoder(
        cached_model, idx, 6, targets, checkpoint_segment_size=1,
    )
    cached_loss.backward()

    assert torch.equal(existing_logits, cached_logits)
    assert torch.equal(existing_loss, cached_loss)
    for name, existing_param in existing_model.named_parameters():
        cached_param = dict(cached_model.named_parameters())[name]
        assert torch.equal(existing_param.grad, cached_param.grad), f"gradient mismatch at {name}"


def test_segment_size_greater_than_one_still_matches_reference():
    reference_model = _model(n_layer=6)
    cached_model = _model(n_layer=6)
    cached_model.load_state_dict(reference_model.state_dict())
    idx = torch.randint(256, (2, 9))
    targets = torch.randint(256, (2, 9))

    _, reference_loss = bdh_variable_depth_forward(reference_model, idx, 6, targets)
    reference_loss.backward()

    _, cached_loss = bdh_wide_gemm_forward_checkpointed_cached_encoder(
        cached_model, idx, 6, targets, checkpoint_segment_size=3,
    )
    cached_loss.backward()

    assert torch.allclose(reference_loss, cached_loss, atol=1e-5, rtol=1e-4)
    for name, ref_param in reference_model.named_parameters():
        ckpt_param = dict(cached_model.named_parameters())[name]
        assert torch.allclose(ref_param.grad, ckpt_param.grad, atol=1e-4, rtol=1e-3), (
            f"gradient mismatch at {name}"
        )


def test_optimizer_step_actually_updates_encoder_and_encoder_v():
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    before_encoder = model.encoder.detach().clone()
    before_encoder_v = model.encoder_v.detach().clone()

    _, loss = bdh_wide_gemm_forward_checkpointed_cached_encoder(model, idx, model.config.n_layer, targets)
    loss.backward()
    optimizer.step()

    assert not torch.equal(before_encoder, model.encoder.detach())
    assert not torch.equal(before_encoder_v, model.encoder_v.detach())


def test_eval_mode_no_grad_matches_plain_wide_gemm_forward():
    from reference.hz0h_bdh_wide_gemm_trainable_torch import bdh_wide_gemm_trainable_forward

    model = _model()
    idx = torch.randint(256, (2, 12))
    with torch.no_grad():
        cached_logits, _ = bdh_wide_gemm_forward_checkpointed_cached_encoder(model, idx, model.config.n_layer)
        plain_logits, _ = bdh_wide_gemm_trainable_forward(model, idx, model.config.n_layer)
    assert torch.equal(cached_logits, plain_logits)
