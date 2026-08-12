"""Regression tests for reference/hz0h_bdh_blocksparse_torch.py's
block_balance_loss -- the real load-balancing auxiliary loss built to
replace the failed exploration_noise fix (docs/restart/hz0h_phase4_blocksparse_results.md
Update 6). Pins down: shape, real gradient flow into model.encoder
(the whole point -- torch.no_grad() would make this a silent no-op),
and the minimum-at-uniform property that motivates the loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_blocksparse_torch import block_balance_loss
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _tiny_config() -> BDHConfig:
    return BDHConfig(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)


def test_returns_a_scalar_tensor():
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDH(config)
    idx = torch.randint(0, config.vocab_size, (4, 10))

    loss = block_balance_loss(model, idx, block_size=4)
    assert loss.dim() == 0


def test_gradient_flows_into_encoder():
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDH(config)
    idx = torch.randint(0, config.vocab_size, (4, 10))

    loss = block_balance_loss(model, idx, block_size=4)
    loss.backward()
    assert model.encoder.grad is not None
    assert model.encoder.grad.abs().sum().item() > 0


def test_minimized_at_uniform_block_scores():
    """Directly check the math (not the model): for a fixed n_blocks,
    a uniform score vector must score lower than a peaked one."""
    n_blocks = 16
    uniform_scores = torch.ones(n_blocks)
    peaked_scores = torch.zeros(n_blocks)
    peaked_scores[0] = 100.0

    def balance_value(scores: torch.Tensor) -> float:
        p = F.softmax(scores, dim=0)
        return (n_blocks * (p * p).sum()).item()

    uniform_loss = balance_value(uniform_scores)
    peaked_loss = balance_value(peaked_scores)
    assert uniform_loss == 1  # exact minimum: n_blocks * (1/n_blocks)^2 * n_blocks = 1
    assert peaked_loss > uniform_loss


def test_loss_decreases_when_encoder_is_optimized_against_it():
    """Real end-to-end sanity check: a few gradient steps minimizing
    ONLY block_balance_loss should actually reduce it (confirms the
    loss is not just differentiable but genuinely optimizable)."""
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDH(config)
    idx = torch.randint(0, config.vocab_size, (4, 10))
    opt = torch.optim.SGD([model.encoder], lr=1.0)

    with torch.no_grad():
        initial_loss = block_balance_loss(model, idx, block_size=4).item()

    for _ in range(20):
        opt.zero_grad()
        loss = block_balance_loss(model, idx, block_size=4)
        loss.backward()
        opt.step()

    with torch.no_grad():
        final_loss = block_balance_loss(model, idx, block_size=4).item()

    assert final_loss < initial_loss
