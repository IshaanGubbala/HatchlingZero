"""Correctness gates for training-only value-path BDH LoRA."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_reasoning_lora_torch import ReasoningLoRABDH, linear_lora_scale
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _config() -> BDHConfig:
    return BDHConfig(n_layer=3, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=64, dropout=0.0)


def test_zero_initialized_reasoning_lora_is_exactly_the_base_oracle():
    config = _config()
    torch.manual_seed(7)
    base = BDH(config).eval()
    torch.manual_seed(11)
    lora = ReasoningLoRABDH(config, rank=4).eval()
    lora.load_base_state_dict(base.state_dict())
    idx = torch.randint(0, config.vocab_size, (2, 9))
    with torch.no_grad():
        base_logits, _ = base(idx)
        lora_logits, _ = lora(idx)
    assert torch.equal(base_logits, lora_logits)


def test_scale_zero_removes_a_nonzero_adapter_exactly_and_merge_preserves_it():
    config = _config()
    model = ReasoningLoRABDH(config, rank=4).eval()
    idx = torch.randint(0, config.vocab_size, (2, 9))
    with torch.no_grad():
        # Make both enabled paths nonzero, then prove removal and merge behavior.
        model.lora_encoder_v_B.normal_(std=0.05)
        model.lora_decoder_B.normal_(std=0.05)
        with_adapter, _ = model(idx)
        model.set_lora_scale(0.0)
        removed, _ = model(idx)
        model.set_lora_scale(1.0)
        model.merge_lora_()
        merged, _ = model(idx)
        model.unmerge_lora_()
        unmerged, _ = model(idx)
    assert not torch.equal(with_adapter, removed)
    assert torch.equal(with_adapter, merged)
    assert torch.equal(with_adapter, unmerged)


def test_adapter_only_mode_freezes_base_and_receives_a_real_gradient():
    config = _config()
    model = ReasoningLoRABDH(config, rank=4, freeze_base=True)
    assert all(not p.requires_grad for n, p in model.named_parameters() if not n.startswith("lora_"))
    assert all(p.requires_grad for n, p in model.named_parameters() if n.startswith("lora_"))
    idx = torch.randint(0, config.vocab_size, (2, 9))
    targets = torch.randint(0, config.vocab_size, (2, 9))
    _, loss = model(idx, targets)
    loss.backward()
    assert model.lora_encoder_v_B.grad is not None
    assert model.lora_decoder_B.grad is not None
    assert model.encoder_v.grad is None
    assert model.decoder.grad is None


def test_default_targets_exclude_addressing_and_schedule_has_exact_endpoints():
    model = ReasoningLoRABDH(_config(), rank=4)
    assert model.targets == frozenset(("encoder_v", "decoder"))
    assert not hasattr(model, "lora_encoder_A")
    assert linear_lora_scale(0, 10, 20) == 1.0
    assert linear_lora_scale(10, 10, 20) == 1.0
    assert linear_lora_scale(15, 10, 20) == 0.5
    assert linear_lora_scale(20, 10, 20) == 0.0


def test_hzcq_value_path_lora_is_exact_at_init_and_changes_hzcq_weights_after_training():
    from reference.hz0h_bdh_hzcq_reasoning_lora_torch import HZCQReasoningLoRA
    from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
    config = BDHVBSubspaceDecoderConfig(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8,
                                        vocab_size=64, dropout=0.0, d_state=8, subspace_rank=4)
    torch.manual_seed(7)
    base = BDHVBSubspaceDecoder(config).eval()
    torch.manual_seed(11)
    lora = HZCQReasoningLoRA(config, rank=2).eval()
    lora.load_base_state_dict(base.state_dict())
    idx = torch.randint(0, 64, (2, 7))
    with torch.no_grad():
        expected, _ = base(idx)
        actual, _ = lora(idx)
        assert torch.equal(expected, actual)
        lora.lora_encoder_v_B.normal_(std=0.1)
        changed, _ = lora(idx)
        lora.merge_lora_()
        merged, _ = lora(idx)
    assert not torch.equal(expected, changed)
    assert torch.equal(changed, merged)
