"""Regression tests for reference/hz0h_bdh_train_torch.py, the verbatim
train.py port built 2026-08-11 as the HZ-0H clean-restart foundation.
Pins down: (1) shifted_target_batch never returns the same tensor twice
and enforces the real x/y slicing, (2) train_step uses the real update
rule and actually reduces loss on learnable (non-random) data, (3) on
genuinely random data (no structure to learn) loss stays at the random
floor rather than dropping -- the same diagnostic that caught the
degenerate same-sequence-target bug in the first place.
"""
from __future__ import annotations

import math

import pytest
import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_train_torch import build_optimizer, shifted_target_batch, train_step


def _tiny_config():
    return BDHConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)


def test_shifted_target_batch_never_aliases_x_and_y():
    full = torch.randint(0, 32, (4, 9))
    x, y = shifted_target_batch(full)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert x.data_ptr() != y.data_ptr()
    assert torch.equal(x, full[:, :-1])
    assert torch.equal(y, full[:, 1:])


def test_shifted_target_batch_rejects_too_short_sequences():
    with pytest.raises(ValueError):
        shifted_target_batch(torch.zeros(2, 1, dtype=torch.long))


def test_train_step_reduces_loss_on_learnable_data():
    """Fixed, repeating token pattern: real structure to learn, so real
    training (the actual update rule from train_step) should reduce loss
    well below the random floor."""
    torch.manual_seed(0)
    config = _tiny_config()
    model = BDH(config)
    opt = build_optimizer(model, lr=2e-3, weight_decay=0.0)

    pattern = torch.arange(0, 9) % 8  # simple repeating structure
    full = pattern.unsqueeze(0).repeat(8, 1)
    x, y = shifted_target_batch(full)

    losses = [train_step(model, opt, x, y) for _ in range(150)]
    random_floor = math.log(config.vocab_size)
    assert losses[-1] < 0.5 * random_floor, f"expected real learning on a learnable pattern, got final loss {losses[-1]} vs floor {random_floor}"


def test_train_step_stays_at_random_floor_on_random_data():
    """No structure to learn -- real training should NOT be able to
    reduce loss below (approximately) the random floor. This is the
    same diagnostic that caught the same-sequence-target bug: a broken
    convention lets the model cheat via the residual stream and shows
    fake 'learning' even here."""
    torch.manual_seed(0)
    config = _tiny_config()
    model = BDH(config)
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.1)
    data_gen = torch.Generator().manual_seed(1234)

    losses = []
    for _ in range(60):
        full = torch.randint(0, config.vocab_size, (4, 9), generator=data_gen)
        x, y = shifted_target_batch(full)
        losses.append(train_step(model, opt, x, y))

    random_floor = math.log(config.vocab_size)
    early = sum(losses[:10]) / 10
    late = sum(losses[-10:]) / 10
    assert abs(late - random_floor) < 0.3, f"loss should stay near the random floor ({random_floor}), got {late}"
    assert abs(late - early) < 0.3, f"loss should not show fake learning on random data, early={early} late={late}"


def test_build_optimizer_uses_real_recipe_defaults():
    config = _tiny_config()
    model = BDH(config)
    opt = build_optimizer(model)
    assert isinstance(opt, torch.optim.AdamW)
    group = opt.param_groups[0]
    assert group["lr"] == pytest.approx(1e-3)
    assert group["weight_decay"] == pytest.approx(0.1)
    n_opt_params = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_model_params = sum(p.numel() for p in model.parameters())
    assert n_opt_params == n_model_params, "real recipe optimizes every parameter, no exclusions"
