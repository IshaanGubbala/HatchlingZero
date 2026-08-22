"""Real correctness tests for
reference/hz0h_bdh_packed_encoder_symmetric_hoisted_rope_torch.py."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_packed_encoder_symmetric_hoisted_rope_torch import (
    bdh_packed_encoder_symmetric_hoisted_rope_forward_checkpointed,
)
from reference.hz0h_bdh_packed_encoder_symmetric_torch import bdh_packed_encoder_symmetric_forward_checkpointed
from reference.hz0h_bdh_packed_encoder_torch import PackedEncoderBDH
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
    candidate_logits, candidate_loss = bdh_packed_encoder_symmetric_hoisted_rope_forward_checkpointed(
        model, idx, model.config.n_layer, targets, checkpoint_segment_size=1,
    )
    assert torch.allclose(reference_logits, candidate_logits, atol=1e-4, rtol=1e-3)
    assert torch.allclose(reference_loss, candidate_loss, atol=1e-4, rtol=1e-3)


def test_matches_non_hoisted_symmetric_forward_bit_exact():
    """The real point: hoisting cos/sin computation must not change the
    math AT ALL, since it's the identical formula just computed once
    instead of per-round -- this should be bit-exact, not just close."""
    config = _config(n_layer=6)
    torch.manual_seed(23)
    non_hoisted_model = PackedEncoderBDH(config)
    torch.manual_seed(23)
    hoisted_model = PackedEncoderBDH(config)
    idx = torch.randint(256, (2, 9))
    targets = torch.randint(256, (2, 9))

    non_hoisted_logits, non_hoisted_loss = bdh_packed_encoder_symmetric_forward_checkpointed(
        non_hoisted_model, idx, 6, targets, checkpoint_segment_size=1,
    )
    non_hoisted_loss.backward()

    hoisted_logits, hoisted_loss = bdh_packed_encoder_symmetric_hoisted_rope_forward_checkpointed(
        hoisted_model, idx, 6, targets, checkpoint_segment_size=1,
    )
    hoisted_loss.backward()

    assert torch.equal(non_hoisted_logits, hoisted_logits)
    assert torch.equal(non_hoisted_loss, hoisted_loss)
    non_hoisted_params = dict(non_hoisted_model.named_parameters())
    hoisted_params = dict(hoisted_model.named_parameters())
    for name in non_hoisted_params:
        ga, gb = non_hoisted_params[name].grad, hoisted_params[name].grad
        assert ga is not None and gb is not None, f"{name} missing a gradient"
        assert torch.equal(ga, gb), f"gradient mismatch at {name}"


def test_one_adamw_step_matches_non_hoisted():
    config = _config(n_layer=6)
    torch.manual_seed(29)
    non_hoisted_model = PackedEncoderBDH(config)
    torch.manual_seed(29)
    hoisted_model = PackedEncoderBDH(config)
    non_hoisted_opt = torch.optim.AdamW(non_hoisted_model.parameters(), lr=3e-4)
    hoisted_opt = torch.optim.AdamW(hoisted_model.parameters(), lr=3e-4)
    idx = torch.randint(256, (2, 9))
    targets = torch.randint(256, (2, 9))

    _, non_hoisted_loss = bdh_packed_encoder_symmetric_forward_checkpointed(non_hoisted_model, idx, 6, targets)
    non_hoisted_loss.backward()
    non_hoisted_opt.step()

    _, hoisted_loss = bdh_packed_encoder_symmetric_hoisted_rope_forward_checkpointed(hoisted_model, idx, 6, targets)
    hoisted_loss.backward()
    hoisted_opt.step()

    non_hoisted_params = dict(non_hoisted_model.named_parameters())
    hoisted_params = dict(hoisted_model.named_parameters())
    for name in non_hoisted_params:
        assert torch.equal(non_hoisted_params[name], hoisted_params[name]), f"post-AdamW mismatch at {name}"


def test_eval_mode_no_grad_matches_oracle():
    config = _config()
    torch.manual_seed(37)
    model = PackedEncoderBDH(config)
    torch.manual_seed(37)
    oracle = BDH(config)
    idx = torch.randint(256, (2, 12))
    with torch.no_grad():
        candidate_logits, _ = bdh_packed_encoder_symmetric_hoisted_rope_forward_checkpointed(model, idx, model.config.n_layer)
        reference_logits, _ = bdh_variable_depth_forward(oracle, idx, oracle.config.n_layer)
    assert torch.allclose(candidate_logits, reference_logits, atol=1e-4, rtol=1e-3)
