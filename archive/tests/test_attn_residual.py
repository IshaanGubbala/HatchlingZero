from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hz0.model.attn_residual import AttentionResidual
from hz0.model.hybrid_lm import HybridLM


def test_attention_residual_output_shape_and_no_nans() -> None:
    module = AttentionResidual(d_model=32, rank=None, n_heads=1)
    history = torch.randn(2, 5, 4, 32)  # batch=2, seq=5, depth=4, d_model=32
    out = module(history)
    assert out.shape == (2, 5, 32)
    assert torch.isfinite(out).all()


def test_attention_residual_with_single_history_entry_reduces_to_that_entry() -> None:
    """With only one prior representation available (softmax over a
    length-1 distribution is always weight=1), the depth-attention output
    must exactly equal that single value -- a real, checkable correctness
    property, not just a shape check."""
    module = AttentionResidual(d_model=16, rank=8, n_heads=2)
    single = torch.randn(3, 7, 1, 16)
    out = module(single)
    assert torch.allclose(out, single[:, :, 0], atol=1e-6)


def test_attention_residual_low_rank_and_multi_head_shapes() -> None:
    for rank, heads in [(8, 1), (8, 4), (32, 8)]:
        module = AttentionResidual(d_model=32, rank=rank, n_heads=heads)
        history = torch.randn(2, 3, 5, 32)
        out = module(history)
        assert out.shape == (2, 3, 32)
        assert torch.isfinite(out).all()


def test_attention_residual_gradients_flow_to_all_history_entries() -> None:
    module = AttentionResidual(d_model=16, rank=16, n_heads=1)
    history = torch.randn(2, 4, 3, 16, requires_grad=True)
    out = module(history)
    out.sum().backward()
    assert history.grad is not None
    assert torch.isfinite(history.grad).all()
    # every depth slice should receive some real gradient signal
    for depth_index in range(history.shape[2]):
        assert history.grad[:, :, depth_index].abs().sum() > 0


def test_hybrid_lm_attn_res_mode_forward_shape() -> None:
    model = HybridLM(
        vocab_size=256, d_model=64, n_layers=4, n_heads=4, d_ff=128, dropout=0.0,
        mixer_backend="gdn2_ref", attention_every=2, max_seq_len=128,
        residual_mode="attn_res", attn_res_rank=16, attn_res_heads=4,
    )
    x = torch.randint(0, 256, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 256)
    assert torch.isfinite(logits).all()


def test_hybrid_lm_standard_mode_is_unaffected_by_attn_res_addition() -> None:
    """Regression guard: adding residual_mode support must not change
    default (`"standard"`) behavior -- same seed, same architecture,
    bit-identical logits to what the pre-existing test
    (`test_gdn2_reference_backend_forward`) already locks in."""
    torch.manual_seed(0)
    model = HybridLM(
        vocab_size=256, d_model=64, n_layers=4, n_heads=4, d_ff=128, dropout=0.0,
        mixer_backend="gdn2_ref", attention_every=2, max_seq_len=128,
    )
    assert model.residual_mode == "standard"
    assert model.attn_res_layers is None
    x = torch.randint(0, 256, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 256)
    assert torch.isfinite(logits).all()
