"""Real correctness tests for
reference/hz0h_bdh_packed_encoder_symmetric_torch.py -- the combined
packed-encoder + symmetric-backward forward.

Tolerance-based (not bit-exact) against both the oracle and the plain
packed-encoder forward, matching the symmetric backward's own established
convention (test_hz0h_bdh_symmetric_backward_torch.py uses atol=1e-5/2e-5
even in plain fp32) -- the (dS+dS.T)@Q identity sums terms in a different
order than generic autograd, so exact floating-point equality was never
the right bar for it."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_packed_encoder_symmetric_torch import bdh_packed_encoder_symmetric_forward_checkpointed
from reference.hz0h_bdh_packed_encoder_torch import PackedEncoderBDH, bdh_packed_encoder_forward_checkpointed
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward


def _config(**overrides) -> BDHConfig:
    return BDHConfig(
        n_layer=overrides.get("n_layer", 5), n_embd=overrides.get("n_embd", 32),
        n_head=overrides.get("n_head", 4), mlp_internal_dim_multiplier=overrides.get("mult", 8),
        vocab_size=256, dropout=0.0,
    )


def test_matches_oracle_logits_and_loss():
    config = _config()
    torch.manual_seed(11)
    model = PackedEncoderBDH(config)
    torch.manual_seed(11)
    oracle = BDH(config)
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    with torch.no_grad():
        reference_logits, reference_loss = bdh_variable_depth_forward(oracle, idx, oracle.config.n_layer, targets)
    combined_logits, combined_loss = bdh_packed_encoder_symmetric_forward_checkpointed(
        model, idx, model.config.n_layer, targets, checkpoint_segment_size=1,
    )
    assert torch.allclose(reference_logits, combined_logits, atol=1e-4, rtol=1e-3)
    assert torch.allclose(reference_loss, combined_loss, atol=1e-4, rtol=1e-3)


def test_matches_oracle_gradients():
    config = _config(n_layer=6)
    torch.manual_seed(23)
    oracle = BDH(config)
    torch.manual_seed(23)
    model = PackedEncoderBDH(config)
    idx = torch.randint(256, (2, 9))
    targets = torch.randint(256, (2, 9))

    _, reference_loss = bdh_variable_depth_forward(oracle, idx, oracle.config.n_layer, targets)
    reference_loss.backward()

    _, combined_loss = bdh_packed_encoder_symmetric_forward_checkpointed(
        model, idx, model.config.n_layer, targets, checkpoint_segment_size=1,
    )
    combined_loss.backward()

    assert torch.allclose(reference_loss, combined_loss, atol=1e-4, rtol=1e-3)
    from reference.hz0h_bdh_packed_encoder_torch import unpack_encoder_view
    nh, D, N = oracle.encoder.shape
    combined_encoder_grad = unpack_encoder_view(model.encoder_packed.grad, nh, N)
    assert torch.allclose(oracle.encoder.grad, combined_encoder_grad, atol=1e-3, rtol=1e-2), (
        f"encoder gradient mismatch: max diff {(oracle.encoder.grad - combined_encoder_grad).abs().max()}"
    )
    oracle_params = dict(oracle.named_parameters())
    model_params = dict(model.named_parameters())
    for name in oracle_params:
        if name == "encoder":
            continue
        ga, gb = oracle_params[name].grad, model_params[name].grad
        assert ga is not None and gb is not None, f"{name} missing a gradient"
        assert torch.allclose(ga, gb, atol=1e-3, rtol=1e-2), f"gradient mismatch at {name}"


def test_close_to_plain_packed_encoder_forward():
    """The two pieces combined must stay close to just the packed-encoder
    forward alone (oracle attention) -- confirms swapping in symmetric
    attention didn't silently change anything beyond its own established
    floating-point-reordering noise."""
    config = _config(n_layer=6)
    torch.manual_seed(31)
    packed_model = PackedEncoderBDH(config)
    torch.manual_seed(31)
    combined_model = PackedEncoderBDH(config)
    idx = torch.randint(256, (2, 9))
    targets = torch.randint(256, (2, 9))

    packed_logits, packed_loss = bdh_packed_encoder_forward_checkpointed(
        packed_model, idx, 6, targets, checkpoint_segment_size=1,
    )
    combined_logits, combined_loss = bdh_packed_encoder_symmetric_forward_checkpointed(
        combined_model, idx, 6, targets, checkpoint_segment_size=1,
    )
    assert torch.allclose(packed_logits, combined_logits, atol=1e-4, rtol=1e-3)
    assert torch.allclose(packed_loss, combined_loss, atol=1e-4, rtol=1e-3)


def test_one_adamw_step_stays_close_to_oracle():
    config = _config(n_layer=6)
    torch.manual_seed(41)
    oracle = BDH(config)
    torch.manual_seed(41)
    model = PackedEncoderBDH(config)
    oracle_opt = torch.optim.AdamW(oracle.parameters(), lr=3e-4)
    model_opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    idx = torch.randint(256, (2, 9))
    targets = torch.randint(256, (2, 9))

    _, oracle_loss = bdh_variable_depth_forward(oracle, idx, 6, targets)
    oracle_loss.backward()
    oracle_opt.step()

    _, combined_loss = bdh_packed_encoder_symmetric_forward_checkpointed(model, idx, 6, targets, checkpoint_segment_size=1)
    combined_loss.backward()
    model_opt.step()

    from reference.hz0h_bdh_packed_encoder_torch import unpack_encoder_view
    nh, D, N = oracle.encoder.shape
    combined_encoder = unpack_encoder_view(model.encoder_packed.detach(), nh, N)
    assert torch.allclose(oracle.encoder.detach(), combined_encoder, atol=1e-3, rtol=1e-2)

    oracle_params = dict(oracle.named_parameters())
    model_params = dict(model.named_parameters())
    for name in oracle_params:
        if name == "encoder":
            continue
        assert torch.allclose(oracle_params[name], model_params[name], atol=1e-3, rtol=1e-2), (
            f"post-AdamW mismatch at {name}"
        )


def test_eval_mode_no_grad_matches_oracle():
    config = _config()
    torch.manual_seed(47)
    model = PackedEncoderBDH(config)
    torch.manual_seed(47)
    oracle = BDH(config)
    idx = torch.randint(256, (2, 12))
    with torch.no_grad():
        combined_logits, _ = bdh_packed_encoder_symmetric_forward_checkpointed(model, idx, model.config.n_layer)
        reference_logits, _ = bdh_variable_depth_forward(oracle, idx, oracle.config.n_layer)
    assert torch.allclose(combined_logits, reference_logits, atol=1e-4, rtol=1e-3)
