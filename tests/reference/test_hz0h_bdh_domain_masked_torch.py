"""Real correctness tests for reference/hz0h_bdh_domain_masked_torch.py.

The load-bearing test is `test_masked_block_receives_exactly_zero_gradient`
-- the entire point of this file is that masking is an EXACT gradient
firewall, not an approximate/soft one, and that claim needs to be
checked, not assumed."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_domain_masked_torch import (
    bdh_domain_masked_forward, build_domain_mask, domain_block_layout,
)
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward

DOMAINS = ["code", "math", "prose", "reasoning", "tools"]


def _model(seed: int = 17, **overrides) -> BDH:
    torch.manual_seed(seed)
    config = BDHConfig(
        n_layer=overrides.get("n_layer", 4), n_embd=overrides.get("n_embd", 32),
        n_head=overrides.get("n_head", 4), mlp_internal_dim_multiplier=overrides.get("mult", 16),
        vocab_size=256, dropout=0.0,
    )
    return BDH(config)


def test_layout_partitions_every_index_exactly_once():
    N = 128
    layout = domain_block_layout(N, DOMAINS, shared_fraction=0.25)
    assert set(layout.keys()) == {"shared", *DOMAINS}
    all_indices = torch.cat(list(layout.values()))
    assert all_indices.numel() == N, f"covered {all_indices.numel()} of {N}"
    assert torch.equal(all_indices.sort().values, torch.arange(N)), "real overlap or gap in the layout"
    # real proportions roughly match the proposal (25% shared, 15% each domain)
    assert layout["shared"].numel() == 32
    for name in DOMAINS[:-1]:
        assert layout[name].numel() == round(N * 0.15)


def test_layout_handles_uneven_division_without_dropping_indices():
    for N in [17, 100, 257, 4992]:
        layout = domain_block_layout(N, DOMAINS, shared_fraction=0.25)
        all_indices = torch.cat(list(layout.values()))
        assert all_indices.numel() == N
        assert torch.equal(all_indices.sort().values, torch.arange(N))


def test_none_mask_matches_oracle_exactly():
    model = _model()
    idx = torch.randint(256, (2, 9))
    targets = torch.randint(256, (2, 9))
    with torch.no_grad():
        ref_logits, ref_loss = bdh_variable_depth_forward(model, idx, model.config.n_layer, targets)
        masked_logits, masked_loss = bdh_domain_masked_forward(model, idx, model.config.n_layer, None, targets)
    assert torch.equal(ref_logits, masked_logits)
    assert torch.equal(ref_loss, masked_loss)


def test_all_ones_mask_matches_oracle_exactly():
    model = _model()
    N = model.config.n_embd * model.config.mlp_internal_dim_multiplier // model.config.n_head
    idx = torch.randint(256, (2, 9))
    targets = torch.randint(256, (2, 9))
    ones_mask = torch.ones(N)
    with torch.no_grad():
        ref_logits, ref_loss = bdh_variable_depth_forward(model, idx, model.config.n_layer, targets)
        masked_logits, masked_loss = bdh_domain_masked_forward(model, idx, model.config.n_layer, ones_mask, targets)
    assert torch.allclose(ref_logits, masked_logits, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ref_loss, masked_loss, atol=1e-6, rtol=1e-5)


def test_masked_block_receives_exactly_zero_gradient():
    """The real, load-bearing property this whole file exists for:
    masking a block must give EXACTLY zero gradient to that block's
    encoder/encoder_v/decoder weight columns/rows -- not approximately
    small, exactly zero, checked bit-for-bit."""
    model = _model(n_embd=64, n_head=4, mult=16)
    N = model.config.n_embd * model.config.mlp_internal_dim_multiplier // model.config.n_head
    layout = domain_block_layout(N, DOMAINS, shared_fraction=0.25)
    mask = build_domain_mask(layout, active_domain="math", n_per_head=N)
    inactive_indices = torch.cat([layout[d] for d in DOMAINS if d != "math"])

    idx = torch.randint(256, (2, 9))
    targets = torch.randint(256, (2, 9))
    _, loss = bdh_domain_masked_forward(model, idx, model.config.n_layer, mask, targets)
    loss.backward()

    # encoder/encoder_v: shape (n_head, D, N) -- inactive columns (last dim)
    assert model.encoder.grad is not None
    assert torch.equal(model.encoder.grad[:, :, inactive_indices], torch.zeros_like(model.encoder.grad[:, :, inactive_indices]))
    assert model.encoder_v.grad is not None
    assert torch.equal(model.encoder_v.grad[:, :, inactive_indices], torch.zeros_like(model.encoder_v.grad[:, :, inactive_indices]))

    # decoder: shape (n_head*N, D) -- inactive ROWS, per head, at offset head*N + n
    nh = model.config.n_head
    inactive_rows = torch.cat([head * N + inactive_indices for head in range(nh)])
    assert model.decoder.grad is not None
    assert torch.equal(model.decoder.grad[inactive_rows, :], torch.zeros_like(model.decoder.grad[inactive_rows, :]))

    # real sanity check: the ACTIVE block did receive real, nonzero gradient
    active_indices = torch.cat([layout["shared"], layout["math"]])
    assert float(model.encoder.grad[:, :, active_indices].abs().sum()) > 0, "active block got zero gradient -- real bug"


def test_optimizer_step_only_moves_active_block_weights():
    """Real end-to-end check: after a real training step, INACTIVE
    block weight columns must be bit-identical to before (zero
    gradient -> zero update), while active columns actually moved."""
    model = _model(n_embd=64, n_head=4, mult=16)
    N = model.config.n_embd * model.config.mlp_internal_dim_multiplier // model.config.n_head
    layout = domain_block_layout(N, DOMAINS, shared_fraction=0.25)
    mask = build_domain_mask(layout, active_domain="code", n_per_head=N)
    inactive_indices = torch.cat([layout[d] for d in DOMAINS if d != "code"])
    active_indices = torch.cat([layout["shared"], layout["code"]])

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    before_encoder = model.encoder.detach().clone()

    idx = torch.randint(256, (2, 9))
    targets = torch.randint(256, (2, 9))
    _, loss = bdh_domain_masked_forward(model, idx, model.config.n_layer, mask, targets)
    loss.backward()
    optimizer.step()

    assert torch.equal(before_encoder[:, :, inactive_indices], model.encoder.detach()[:, :, inactive_indices])
    assert not torch.equal(before_encoder[:, :, active_indices], model.encoder.detach()[:, :, active_indices])


def test_masking_a_different_domain_isolates_a_different_block():
    """Real, direct confirmation that masking follows the LABEL, not a
    fixed block -- switching domain moves which columns are frozen."""
    model = _model(n_embd=64, n_head=4, mult=16)
    N = model.config.n_embd * model.config.mlp_internal_dim_multiplier // model.config.n_head
    layout = domain_block_layout(N, DOMAINS, shared_fraction=0.25)
    idx = torch.randint(256, (2, 9))
    targets = torch.randint(256, (2, 9))

    for active_domain in ["code", "reasoning"]:
        model.zero_grad(set_to_none=True)
        mask = build_domain_mask(layout, active_domain=active_domain, n_per_head=N)
        _, loss = bdh_domain_masked_forward(model, idx, model.config.n_layer, mask, targets)
        loss.backward()
        for other in DOMAINS:
            if other == active_domain:
                continue
            other_indices = layout[other]
            assert torch.equal(
                model.encoder.grad[:, :, other_indices], torch.zeros_like(model.encoder.grad[:, :, other_indices])
            ), f"domain={active_domain}: block {other} unexpectedly got gradient"
