"""Forward and backward parity for the exact N-dominant BDH scan."""
from __future__ import annotations

import pytest
import torch

from reference.hz0h_bdh_native_kernel_attention_torch import (
    bdh_native_attention,
    bdh_native_forward,
)
from reference.hz0h_bdh_torch import Attention, BDH, BDHConfig


def _freqs(n: int) -> torch.Tensor:
    return Attention(BDHConfig(n_embd=16, n_head=2, mlp_internal_dim_multiplier=n // 8)).freqs


@pytest.mark.parametrize(
    "seed,batch,heads,tokens,width,value_width,chunk_size",
    [(0, 1, 2, 1, 8, 4, 3), (1, 2, 2, 7, 16, 8, 3), (2, 1, 4, 9, 24, 6, 4), (3, 2, 3, 11, 32, 5, 5)],
)
def test_native_attention_matches_oracle_forward_and_gradients(
    seed, batch, heads, tokens, width, value_width, chunk_size
):
    torch.manual_seed(seed)
    freqs = _freqs(width).expand(1, 1, 1, width)
    q_oracle = torch.randn(batch, heads, tokens, width, requires_grad=True)
    v_oracle = torch.randn(batch, 1, tokens, value_width, requires_grad=True)
    q_native = q_oracle.detach().clone().requires_grad_()
    v_native = v_oracle.detach().clone().requires_grad_()

    oracle = Attention.rope(
        torch.arange(tokens, dtype=freqs.dtype).view(1, 1, tokens, 1) * freqs,
        q_oracle,
    )
    oracle = (oracle @ oracle.mT).tril(diagonal=-1) @ v_oracle
    native = bdh_native_attention(q_native, v_native, freqs, query_chunk_size=chunk_size)
    assert torch.allclose(native, oracle, atol=1e-3, rtol=1e-3)

    loss_oracle = (oracle.square().mean() + oracle.sum() * 0.01)
    loss_native = (native.square().mean() + native.sum() * 0.01)
    loss_oracle.backward()
    loss_native.backward()
    assert torch.allclose(q_native.grad, q_oracle.grad, atol=1e-2, rtol=1e-2)
    assert torch.allclose(v_native.grad, v_oracle.grad, atol=1e-2, rtol=1e-2)


def test_native_model_forward_and_all_parameter_gradients_match():
    config = BDHConfig(
        n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8,
        vocab_size=32, dropout=0.0,
    )
    torch.manual_seed(11)
    oracle_model = BDH(config)
    native_model = BDH(config)
    native_model.load_state_dict(oracle_model.state_dict())
    idx = torch.randint(0, config.vocab_size, (2, 9))
    targets = torch.randint(0, config.vocab_size, idx.shape)

    oracle_logits, oracle_loss = oracle_model(idx, targets)
    native_logits, native_loss = bdh_native_forward(native_model, idx, targets)
    assert torch.allclose(native_logits, oracle_logits, atol=1e-3, rtol=1e-3)
    assert torch.allclose(native_loss, oracle_loss, atol=1e-3, rtol=1e-3)
    oracle_loss.backward()
    native_loss.backward()

    for (name_a, param_a), (name_b, param_b) in zip(
        oracle_model.named_parameters(), native_model.named_parameters()
    ):
        assert name_a == name_b
        assert param_a.grad is not None and param_b.grad is not None
        assert torch.allclose(param_a.grad, param_b.grad, atol=1e-2, rtol=1e-2), name_a
