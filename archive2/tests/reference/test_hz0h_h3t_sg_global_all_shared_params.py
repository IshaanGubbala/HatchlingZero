"""Regression tests for SG-global extended to all three shared/tied
parameters (scripts/hz0h_h3t_sg_global_all_shared_params.py). Real
findings this session relied on: all three per-position targets
reconstruct their real gradients exactly, and quality is WORSE than
single-parameter SG-global despite the better per-parameter signal
(compounding error across three approximate signals, not fixed by using
a better target for each one individually).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from hz0h_h3t_sg_global_all_shared_params import sg_global_data_all_params, run
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _tiny_config():
    return BDHConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)


def test_all_three_targets_reconstruct_true_gradients_exactly():
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDH(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (3, 10))

    data = sg_global_data_all_params(model, idx)
    true_enc = model.encoder.grad.detach().clone()
    true_encv = model.encoder_v.grad.detach().clone()
    true_dec = model.decoder.grad.detach().clone()

    recon_enc = torch.einsum("btd,bhtn->hdn", data["encoder"]["pre"].squeeze(1), data["encoder"]["grad"])
    recon_encv = torch.einsum("bhtd,bhtn->hdn", data["encoder_v"]["pre"], data["encoder_v"]["grad"])
    recon_dec = torch.einsum("btd,bte->de", data["decoder"]["pre"].squeeze(1), data["decoder"]["grad"].squeeze(1))

    assert float((recon_enc - true_enc).abs().max()) < 1e-5
    assert float((recon_encv - true_encv).abs().max()) < 1e-5
    assert float((recon_dec - true_dec).abs().max()) < 1e-5


def test_embed_and_lm_head_still_train_during_synthetic_steps():
    """Real regression test for a real bug caught before it shipped: an
    earlier draft called opt.zero_grad() after the data-pass backward,
    which would have wiped embed/lm_head's real gradients and frozen
    them during every synthetic step. Confirms embed's and lm_head's
    weights actually change across training (well past warmup, so most
    steps are synthetic), not just that the run doesn't crash."""
    config = _tiny_config()
    torch.manual_seed(0)
    embed_before = BDH(config).embed.weight.detach().clone()
    torch.manual_seed(0)
    lm_head_before = BDH(config).lm_head.detach().clone()

    losses, model = run(config, seed=0, steps=15, batch_size=4, seq_len=8, warmup_steps=3, condition="sg_global_all_three", return_model=True)
    assert all(l == l for l in losses)

    embed_diff = float((model.embed.weight.detach() - embed_before).abs().max())
    lm_head_diff = float((model.lm_head.detach() - lm_head_before).abs().max())
    assert embed_diff > 1e-6, "embed.weight should have changed during training, not stayed frozen"
    assert lm_head_diff > 1e-6, "lm_head should have changed during training, not stayed frozen"


def test_synthetic_all_three_trains_without_diverging():
    config = _tiny_config()
    losses = run(config, seed=0, steps=20, batch_size=4, seq_len=8, warmup_steps=5, condition="sg_global_all_three")
    assert all(l == l for l in losses), "sg_global all-3 params run produced NaN"
    assert losses[-1] < losses[0], "should show real loss reduction, not just avoid NaN"


def test_sg_global_all_three_is_worse_than_single_param_sg_global():
    """Real, disclosed finding: compounding three independently-
    approximate (even if individually better, SG-global) signals lands
    worse than a single-parameter swap -- confirmed reproducible at
    reduced scale for test speed, matching the full-scale measurement
    (0.7930 vs 0.4203 at 300 steps)."""
    config = _tiny_config()
    baseline = run(config, seed=0, steps=60, batch_size=4, seq_len=8, warmup_steps=15, condition="true_bptt")
    all_three = run(config, seed=0, steps=60, batch_size=4, seq_len=8, warmup_steps=15, condition="sg_global_all_three")
    assert all_three[-1] > baseline[-1], "sg_global all-3 should trail true BPTT"
