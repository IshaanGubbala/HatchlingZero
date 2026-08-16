"""CUDA correctness gate for the hand-written Triton BDH attention kernel."""
from __future__ import annotations

import pytest
import torch

from reference.hz0h_bdh_torch import Attention, BDHConfig
from reference.hz0h_bdh_triton_attention_torch import (
    bdh_triton_attention,
    triton_available,
)


pytestmark = pytest.mark.skipif(
    not triton_available(), reason="requires CUDA and Triton"
)


def test_triton_forward_and_gradients_match_oracle():
    config = BDHConfig(n_embd=32, n_head=4, mlp_internal_dim_multiplier=8)
    attention = Attention(config).to(device="cuda", dtype=torch.bfloat16)
    attention.freqs = attention.freqs.to(torch.float32)
    torch.manual_seed(19)
    q_oracle = torch.randn(2, 4, 17, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v_oracle = torch.randn(2, 1, 17, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q_kernel = q_oracle.detach().clone().requires_grad_()
    v_kernel = v_oracle.detach().clone().requires_grad_()
    oracle = attention(q_oracle, q_oracle, v_oracle)
    kernel = bdh_triton_attention(q_kernel, v_kernel, attention.freqs)
    assert torch.allclose(kernel, oracle, atol=1e-2, rtol=1e-2)
    oracle.square().mean().backward()
    kernel.square().mean().backward()
    assert torch.allclose(q_kernel.grad, q_oracle.grad, atol=2e-2, rtol=2e-2)
    assert torch.allclose(v_kernel.grad, v_oracle.grad, atol=2e-2, rtol=2e-2)
