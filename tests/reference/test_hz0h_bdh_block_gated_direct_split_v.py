from __future__ import annotations

import copy
import torch

from reference.hz0h_bdh_block_gated_torch import (
    BDHBlockGated, BDHBlockGatedConfig, bdh_block_gated_annealed_direct_split_v_chunk_gla_forward, bdh_block_gated_annealed_direct_split_v_compact_gate_forward, bdh_block_gated_annealed_direct_split_v_forward,
    bdh_block_gated_forward,
)


def _model():
    return BDHBlockGated(BDHBlockGatedConfig(n_layer=2, n_embd=32, n_head=4,
        mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0, block_size=4))


def test_dense_gate_phase_remains_exact_established_gate_forward():
    torch.manual_seed(9)
    model = _model().eval()
    idx = torch.randint(0, 32, (2, 9))
    expected, _ = bdh_block_gated_forward(model, idx)
    actual, _ = bdh_block_gated_annealed_direct_split_v_forward(model, idx, 1.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sparse_direct_value_gate_path_has_finite_gradients_including_gate():
    torch.manual_seed(10)
    model = _model()
    idx = torch.randint(0, 32, (2, 9))
    logits, loss = bdh_block_gated_annealed_direct_split_v_forward(model, idx, 0.5, targets=idx)
    assert logits.shape == (2, 9, 32)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.gate.grad is not None and torch.isfinite(model.gate.grad).all()
    assert float(model.gate.grad.norm()) > 0
    assert model.encoder.grad is not None and torch.isfinite(model.encoder.grad).all()


def test_gated_chunk_gla_refuses_non_cuda():
    model = _model()
    idx = torch.randint(0, 32, (1, 8))
    try:
        bdh_block_gated_annealed_direct_split_v_chunk_gla_forward(model, idx, 0.5)
        assert False, "expected CUDA requirement"
    except RuntimeError as exc:
        assert "CUDA/Triton" in str(exc)


def test_compact_gate_matches_legacy_logits_loss_and_gradients():
    torch.manual_seed(22)
    legacy = _model()
    compact = copy.deepcopy(legacy)
    idx = torch.randint(0, 32, (2, 9))
    legacy_logits, legacy_loss = bdh_block_gated_annealed_direct_split_v_forward(legacy, idx, 0.5, targets=idx)
    compact_logits, compact_loss = bdh_block_gated_annealed_direct_split_v_compact_gate_forward(compact, idx, 0.5, targets=idx)
    torch.testing.assert_close(compact_logits, legacy_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(compact_loss, legacy_loss, rtol=1e-5, atol=1e-6)
    legacy_loss.backward(); compact_loss.backward()
    for (name, old), (_, new) in zip(legacy.named_parameters(), compact.named_parameters()):
        assert name
        torch.testing.assert_close(new.grad, old.grad, rtol=1e-4, atol=1e-5)
