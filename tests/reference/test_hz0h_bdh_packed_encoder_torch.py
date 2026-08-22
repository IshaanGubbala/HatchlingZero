"""Real correctness tests for reference/hz0h_bdh_packed_encoder_torch.py.

Load-bearing gates: (1) PackedEncoderBDH's initial weights must be bit-
identical to a same-seeded oracle BDH once converted back, since del +
re-register could silently reinitialize or misalign something; (2) logits/
loss/gradients/AdamW updates must match the existing checkpointed forward
exactly through the unpack conversion."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_packed_encoder_torch import (
    PackedEncoderBDH,
    bdh_packed_encoder_forward_checkpointed,
    unpack_encoder_view,
)
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward
from reference.hz0h_bdh_wide_gemm_checkpointed_torch import bdh_wide_gemm_forward_checkpointed


def _config(**overrides) -> BDHConfig:
    return BDHConfig(
        n_layer=overrides.get("n_layer", 5), n_embd=overrides.get("n_embd", 32),
        n_head=overrides.get("n_head", 4), mlp_internal_dim_multiplier=overrides.get("mult", 8),
        vocab_size=256, dropout=0.0,
    )


def test_packed_construction_preserves_every_initial_weight():
    """Same seed, same __init__ code path -- del self.encoder + re-register
    as encoder_packed must not perturb ANY parameter, including the ones
    never touched (embed, attn, decoder, encoder_v, lm_head)."""
    config = _config()
    torch.manual_seed(17)
    oracle = BDH(config)
    torch.manual_seed(17)
    packed = PackedEncoderBDH(config)

    nh, D, N = oracle.encoder.shape
    assert torch.equal(unpack_encoder_view(packed.encoder_packed, nh, N), oracle.encoder)

    oracle_params = dict(oracle.named_parameters())
    packed_params = dict(packed.named_parameters())
    assert set(oracle_params) - {"encoder"} == set(packed_params) - {"encoder_packed"}
    for name in oracle_params:
        if name == "encoder":
            continue
        assert torch.equal(oracle_params[name], packed_params[name]), f"mismatch at {name}"


def test_matches_reference_logits_exactly():
    config = _config()
    torch.manual_seed(11)
    model = PackedEncoderBDH(config)
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    torch.manual_seed(11)
    oracle = BDH(config)
    with torch.no_grad():
        reference_logits, reference_loss = bdh_variable_depth_forward(oracle, idx, oracle.config.n_layer, targets)
    packed_logits, packed_loss = bdh_packed_encoder_forward_checkpointed(
        model, idx, model.config.n_layer, targets, checkpoint_segment_size=1,
    )
    assert torch.allclose(reference_logits, packed_logits, atol=1e-5, rtol=1e-4)
    assert torch.allclose(reference_loss, packed_loss, atol=1e-5, rtol=1e-4)


def test_matches_existing_checkpointed_forward_gradients_through_unpack():
    """The real point: existing checkpointed forward's model.encoder.grad,
    once converted through the same permute/reshape unpack, must equal
    PackedEncoderBDH's model.encoder_packed.grad exactly -- proving the
    packed layout isn't just forward-equivalent but backward-equivalent
    too."""
    config = _config(n_layer=6)
    torch.manual_seed(23)
    existing_model = BDH(config)
    torch.manual_seed(23)
    packed_model = PackedEncoderBDH(config)

    idx = torch.randint(256, (2, 9))
    targets = torch.randint(256, (2, 9))

    existing_logits, existing_loss = bdh_wide_gemm_forward_checkpointed(
        existing_model, idx, 6, targets, checkpoint_segment_size=1,
    )
    existing_loss.backward()

    packed_logits, packed_loss = bdh_packed_encoder_forward_checkpointed(
        packed_model, idx, 6, targets, checkpoint_segment_size=1,
    )
    packed_loss.backward()

    assert torch.allclose(existing_logits, packed_logits, atol=1e-5, rtol=1e-4)
    assert torch.allclose(existing_loss, packed_loss, atol=1e-5, rtol=1e-4)

    nh, D, N = existing_model.encoder.shape
    packed_grad_unpacked = unpack_encoder_view(packed_model.encoder_packed.grad, nh, N)
    assert torch.allclose(existing_model.encoder.grad, packed_grad_unpacked, atol=1e-4, rtol=1e-3), (
        f"encoder gradient mismatch: max diff {(existing_model.encoder.grad - packed_grad_unpacked).abs().max()}"
    )

    existing_params = dict(existing_model.named_parameters())
    packed_params = dict(packed_model.named_parameters())
    for name in existing_params:
        if name == "encoder":
            continue
        ga, gb = existing_params[name].grad, packed_params[name].grad
        assert ga is not None and gb is not None, f"{name} missing a gradient"
        assert torch.allclose(ga, gb, atol=1e-4, rtol=1e-3), f"gradient mismatch at {name}"


def test_one_adamw_step_matches_existing_checkpointed_forward_through_unpack():
    config = _config(n_layer=6)
    torch.manual_seed(29)
    existing_model = BDH(config)
    torch.manual_seed(29)
    packed_model = PackedEncoderBDH(config)

    existing_opt = torch.optim.AdamW(existing_model.parameters(), lr=3e-4)
    packed_opt = torch.optim.AdamW(packed_model.parameters(), lr=3e-4)

    idx = torch.randint(256, (2, 9))
    targets = torch.randint(256, (2, 9))

    _, existing_loss = bdh_wide_gemm_forward_checkpointed(existing_model, idx, 6, targets, checkpoint_segment_size=1)
    existing_loss.backward()
    existing_opt.step()

    _, packed_loss = bdh_packed_encoder_forward_checkpointed(packed_model, idx, 6, targets, checkpoint_segment_size=1)
    packed_loss.backward()
    packed_opt.step()

    nh, D, N = existing_model.encoder.shape
    packed_encoder_unpacked = unpack_encoder_view(packed_model.encoder_packed.detach(), nh, N)
    assert torch.allclose(existing_model.encoder.detach(), packed_encoder_unpacked, atol=1e-5, rtol=1e-4)

    existing_params = dict(existing_model.named_parameters())
    packed_params = dict(packed_model.named_parameters())
    for name in existing_params:
        if name == "encoder":
            continue
        assert torch.allclose(existing_params[name], packed_params[name], atol=1e-5, rtol=1e-4), (
            f"post-AdamW mismatch at {name}"
        )


def test_segment_size_greater_than_one_still_matches_reference():
    config = _config(n_layer=6)
    torch.manual_seed(31)
    model = PackedEncoderBDH(config)
    torch.manual_seed(31)
    oracle = BDH(config)
    idx = torch.randint(256, (2, 9))
    targets = torch.randint(256, (2, 9))

    with torch.no_grad():
        reference_logits, reference_loss = bdh_variable_depth_forward(oracle, idx, 6, targets)

    packed_logits, packed_loss = bdh_packed_encoder_forward_checkpointed(
        model, idx, 6, targets, checkpoint_segment_size=3,
    )
    assert torch.allclose(reference_logits, packed_logits, atol=1e-5, rtol=1e-4)
    assert torch.allclose(reference_loss, packed_loss, atol=1e-5, rtol=1e-4)


def test_eval_mode_no_grad_matches_reference():
    config = _config()
    torch.manual_seed(37)
    model = PackedEncoderBDH(config)
    torch.manual_seed(37)
    oracle = BDH(config)
    idx = torch.randint(256, (2, 12))
    with torch.no_grad():
        packed_logits, _ = bdh_packed_encoder_forward_checkpointed(model, idx, model.config.n_layer)
        reference_logits, _ = bdh_variable_depth_forward(oracle, idx, oracle.config.n_layer)
    assert torch.allclose(packed_logits, reference_logits, atol=1e-5, rtol=1e-4)
