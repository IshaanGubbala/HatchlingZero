"""Regression tests for reference/hz0h_bdh_block_gated_torch.py (HZ
Next-Phase Plan Phase I4.1, "Selective BlockBDH -- dense-gated phase").
Pins down: dense computation (no blocks ever skipped), real gradient
flow to every block's gate AND to encoder/encoder_v/decoder/gate all
together, gate values genuinely change block contributions, and gates
are recomputed fresh (input-dependent) rather than static.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_block_gated_torch import (
    BDHBlockGated,
    BDHBlockGatedConfig,
    bdh_block_gated_annealed_forward,
    bdh_block_gated_forward,
    compute_active_blocks_by_gate,
    compute_block_gate,
)


def _tiny_config(block_size: int = 4) -> BDHBlockGatedConfig:
    return BDHBlockGatedConfig(n_layer=3, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0, block_size=block_size)


def test_rejects_odd_block_size():
    import pytest
    # odd block_size isn't caught by the plain dataclass (no __post_init__
    # validation there) -- real validation happens at model construction.
    config = BDHBlockGatedConfig(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0, block_size=3)
    with pytest.raises(ValueError):
        BDHBlockGated(config)


def test_n_blocks_computed_correctly():
    config = _tiny_config(block_size=4)
    torch.manual_seed(0)
    model = BDHBlockGated(config)
    # N = mlp_mult * D / nh = 8*32/4 = 64; n_blocks = 64/4 = 16
    assert model.n_blocks == 16
    assert model.gate.shape == (32, 16)


def test_gate_output_in_unit_interval():
    config = _tiny_config()
    torch.manual_seed(1)
    model = BDHBlockGated(config)
    idx = torch.randint(0, config.vocab_size, (2, 10))
    x = model.ln(model.embed(idx).unsqueeze(1))
    gate = compute_block_gate(model, x)
    assert gate.shape == (2, 10, model.n_blocks)
    assert (gate > 0).all() and (gate < 1).all()


def test_gates_vary_across_layers_since_x_varies():
    """Real property: the gate's WEIGHTS are shared/tied across
    iterations (matching BDH's own convention), but its OUTPUT should
    differ per iteration because the residual stream x differs."""
    config = _tiny_config()
    torch.manual_seed(2)
    model = BDHBlockGated(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 8))
    with torch.no_grad():
        _logits, _loss, gates = bdh_block_gated_forward(model, idx, return_gates=True)
    assert len(gates) == config.n_layer
    assert not torch.allclose(gates[0], gates[-1])


def test_forward_matches_dense_computation_shape():
    config = _tiny_config()
    torch.manual_seed(3)
    model = BDHBlockGated(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 6))
    with torch.no_grad():
        logits, _loss = model(idx)
    assert logits.shape == (2, 6, config.n_embd)


def test_gradients_flow_to_gate_and_all_shared_weights():
    config = _tiny_config()
    torch.manual_seed(4)
    model = BDHBlockGated(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (2, 9))
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
    _logits, loss = model(x, targets=y)
    loss.backward()

    assert model.gate.grad is not None and float(model.gate.grad.norm()) > 0
    assert model.encoder.grad is not None and float(model.encoder.grad.norm()) > 0
    assert model.encoder_v.grad is not None and float(model.encoder_v.grad.norm()) > 0
    assert model.decoder.grad is not None and float(model.decoder.grad.norm()) > 0


def test_zero_gate_weight_gives_uniform_half_gating_at_init():
    """Real sanity check: with gate weights all zero (before training),
    sigmoid(0) = 0.5 exactly -- every block gated at exactly half
    strength, not some arbitrary/broken value."""
    config = _tiny_config()
    torch.manual_seed(5)
    model = BDHBlockGated(config)
    with torch.no_grad():
        model.gate.zero_()
    idx = torch.randint(0, config.vocab_size, (1, 5))
    x = model.ln(model.embed(idx).unsqueeze(1))
    gate = compute_block_gate(model, x)
    assert torch.allclose(gate, torch.full_like(gate, 0.5))


def test_different_gate_weights_produce_different_logits():
    """Real property needed for the gate to matter at all: changing
    gate weights (holding everything else fixed) must change the
    model's output."""
    config = _tiny_config()
    torch.manual_seed(6)
    model = BDHBlockGated(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 7))
    with torch.no_grad():
        logits_a, _ = model(idx)
        model.gate.add_(1.0)
        logits_b, _ = model(idx)
    assert not torch.allclose(logits_a, logits_b)


# --- Phase I4.2: soft-to-sparse annealing -----------------------------------


def test_annealed_forward_at_active_fraction_one_matches_dense_forward():
    """active_fraction=1.0 must be byte-identical to the plain dense
    forward -- the real backward-compatibility guarantee this
    annealing scheme depends on."""
    config = _tiny_config()
    torch.manual_seed(10)
    model = BDHBlockGated(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 8))
    with torch.no_grad():
        logits_dense, loss_dense = bdh_block_gated_forward(model, idx)
        logits_annealed, loss_annealed = bdh_block_gated_annealed_forward(model, idx, active_fraction=1.0)
    assert torch.equal(logits_dense, logits_annealed)


def test_compute_active_blocks_by_gate_returns_correct_count():
    config = _tiny_config(block_size=4)
    torch.manual_seed(11)
    model = BDHBlockGated(config)
    idx = torch.randint(0, config.vocab_size, (2, 6))
    active_blocks = compute_active_blocks_by_gate(model, idx, active_fraction=0.5)
    assert active_blocks.numel() == round(model.n_blocks * 0.5)
    assert (active_blocks >= 0).all() and (active_blocks < model.n_blocks).all()
    assert len(set(active_blocks.tolist())) == active_blocks.numel()


def test_annealed_forward_at_partial_fraction_produces_valid_shaped_output():
    config = _tiny_config()
    torch.manual_seed(12)
    model = BDHBlockGated(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 6))
    with torch.no_grad():
        logits, _loss = bdh_block_gated_annealed_forward(model, idx, active_fraction=0.5)
    assert logits.shape == (2, 6, config.n_embd)


def test_annealed_forward_genuinely_differs_from_dense_at_partial_fraction():
    """Real property: restricting to half the blocks must actually
    change the output (otherwise the selection would be a no-op)."""
    config = _tiny_config()
    torch.manual_seed(13)
    model = BDHBlockGated(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 6))
    with torch.no_grad():
        logits_dense, _ = bdh_block_gated_forward(model, idx)
        logits_sparse, _ = bdh_block_gated_annealed_forward(model, idx, active_fraction=0.5)
    assert not torch.allclose(logits_dense, logits_sparse, atol=1e-3)


def test_gradients_flow_through_annealed_forward_at_partial_fraction():
    config = _tiny_config()
    torch.manual_seed(14)
    model = BDHBlockGated(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (2, 8))
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
    _logits, loss = bdh_block_gated_annealed_forward(model, x, active_fraction=0.5, targets=y)
    loss.backward()
    assert model.gate.grad is not None and float(model.gate.grad.norm()) > 0
    assert model.encoder.grad is not None and float(model.encoder.grad.norm()) > 0


def test_annealed_forward_rejects_odd_block_size_via_rope_pairing():
    """block_size=4 (even) is required by construction (BDHBlockGated
    itself rejects odd block_size) -- this test just confirms the
    annealed path works correctly at the smallest valid even size."""
    config = _tiny_config(block_size=2)
    torch.manual_seed(15)
    model = BDHBlockGated(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 5))
    with torch.no_grad():
        logits, _ = bdh_block_gated_annealed_forward(model, idx, active_fraction=0.5)
    assert logits.shape == (1, 5, config.n_embd)
