"""Real correctness tests for reference/hz0h_bdh_depth_adapter_torch.py.

Load-bearing gate: at construction, `B_group` is zero, so every group's
`W_shared + A_group @ B_group` must equal `W_shared` EXACTLY -- meaning
`AdapterDepthBDH` at init must reproduce the real oracle bit-for-bit,
same gate pattern as `DepthUntiedBDH`."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_depth_adapter_torch import AdapterDepthBDH
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _config(**overrides) -> BDHConfig:
    return BDHConfig(
        n_layer=overrides.get("n_layer", 4), n_embd=overrides.get("n_embd", 32),
        n_head=overrides.get("n_head", 4), mlp_internal_dim_multiplier=overrides.get("mult", 8),
        vocab_size=256, dropout=0.0,
    )


def test_zero_init_adapters_reproduce_the_real_oracle_exactly():
    config = _config()
    torch.manual_seed(7)
    oracle = BDH(config).eval()
    torch.manual_seed(11)  # different seed on purpose -- A's random init must not matter while B is zero
    adapted = AdapterDepthBDH(config, depth=config.n_layer, groups=2, rank=4).eval()
    adapted.shared_encoder.data.copy_(oracle.encoder.data)
    adapted.shared_encoder_v.data.copy_(oracle.encoder_v.data)
    adapted.shared_decoder.data.copy_(oracle.decoder.data)
    adapted.embed.load_state_dict(oracle.embed.state_dict())
    adapted.lm_head.data.copy_(oracle.lm_head.data)

    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    with torch.no_grad():
        oracle_logits, oracle_loss = oracle(idx, targets)
        adapted_logits, adapted_loss = adapted(idx, targets)
    assert torch.equal(oracle_logits, adapted_logits), "zero-init adapter forward is not the real oracle"
    assert torch.equal(oracle_loss, adapted_loss)


def test_encoder_for_is_exactly_shared_at_init():
    config = _config()
    torch.manual_seed(7)
    model = AdapterDepthBDH(config, depth=4, groups=2, rank=4)
    for group in range(model.groups):
        assert torch.equal(model.encoder_for(group), model.shared_encoder)
        assert torch.equal(model.encoder_v_for(group), model.shared_encoder_v)
        assert torch.equal(model.decoder_for(group), model.shared_decoder)


def test_training_step_makes_groups_diverge_and_gradients_flow_to_both_shared_and_adapters():
    """Real structural check: after one real gradient step, different
    groups' effective weights must differ (the adapters are doing
    something), and BOTH the shared base and the per-group A/B factors
    must receive gradients (neither is dead)."""
    config = _config()
    torch.manual_seed(7)
    model = AdapterDepthBDH(config, depth=4, groups=2, rank=4)
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))

    _, loss = model(idx, targets)
    loss.backward()
    assert model.shared_encoder.grad is not None and torch.isfinite(model.shared_encoder.grad).all()
    for group in range(model.groups):
        assert model.enc_A[group].grad is not None
        assert model.enc_B[group].grad is not None
        assert torch.isfinite(model.enc_A[group].grad).all()
        assert torch.isfinite(model.enc_B[group].grad).all()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()

    with torch.no_grad():
        assert not torch.equal(model.encoder_for(0), model.encoder_for(1)), (
            "groups must diverge after a real gradient step"
        )


def test_adapter_parameter_count_is_much_smaller_than_full_untying():
    """Real sanity check on the whole point of this file: the adapter's
    extra parameters (A/B factors only) must be a small fraction of what
    DepthUntiedBDH would spend giving every group a full-size matrix."""
    config = _config(n_embd=256, n_head=4, mult=16)
    groups = 8
    rank = 8
    model = AdapterDepthBDH(config, depth=8, groups=groups, rank=rank)
    nh, D = config.n_head, config.n_embd
    N = D * config.mlp_internal_dim_multiplier // nh
    full_untied_extra_params = groups * (2 * nh * D * N + nh * N * D)  # encoder+encoder_v+decoder per group
    assert model.adapter_parameter_count() < full_untied_extra_params * 0.1
