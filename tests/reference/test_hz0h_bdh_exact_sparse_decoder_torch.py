from __future__ import annotations

import pytest
import torch

from reference.hz0h_bdh_exact_sparse_decoder_torch import (
    bdh_exact_sparse_decoder_forward,
    exact_sparse_decoder_mm,
)
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_wide_gemm_trainable_torch import bdh_wide_gemm_trainable_forward


def _model(seed: int = 19) -> BDH:
    torch.manual_seed(seed)
    return BDH(
        BDHConfig(
            n_layer=3,
            n_embd=24,
            n_head=4,
            mlp_internal_dim_multiplier=8,
            vocab_size=256,
            dropout=0.0,
        )
    )


@pytest.mark.parametrize("layout", ["coo", "csr"])
def test_sparse_decoder_operator_matches_dense_forward_and_backward(layout: str):
    torch.manual_seed(3)
    dense_input = torch.randn(7, 32)
    dense_input[dense_input < 0.8] = 0
    dense_weight = torch.randn(32, 11)

    reference_input = dense_input.clone().requires_grad_(True)
    reference_weight = dense_weight.clone().requires_grad_(True)
    reference = reference_input @ reference_weight
    reference.square().sum().backward()

    sparse_input = dense_input.clone().requires_grad_(True)
    sparse_weight = dense_weight.clone().requires_grad_(True)
    candidate = exact_sparse_decoder_mm(sparse_input, sparse_weight, layout=layout)
    candidate.square().sum().backward()

    assert torch.allclose(candidate, reference, atol=1e-6, rtol=1e-6)
    active = dense_input != 0
    assert torch.allclose(
        sparse_input.grad[active], reference_input.grad[active], atol=1e-6, rtol=1e-6
    )
    assert torch.count_nonzero(sparse_input.grad[~active]) == 0
    assert torch.allclose(sparse_weight.grad, reference_weight.grad, atol=1e-6, rtol=1e-6)


def test_full_sparse_decoder_path_matches_every_parameter_gradient():
    reference_model = _model()
    sparse_model = _model()
    sparse_model.load_state_dict(reference_model.state_dict())
    idx = torch.randint(256, (2, 10))
    targets = torch.randint(256, (2, 10))

    reference_logits, reference_loss = bdh_wide_gemm_trainable_forward(
        reference_model, idx, reference_model.config.n_layer, targets
    )
    sparse_logits, sparse_loss = bdh_exact_sparse_decoder_forward(
        sparse_model, idx, sparse_model.config.n_layer, targets, sparse_layout="coo"
    )
    reference_loss.backward()
    sparse_loss.backward()

    assert torch.allclose(sparse_logits, reference_logits, atol=1e-5, rtol=1e-4)
    assert torch.allclose(sparse_loss, reference_loss, atol=1e-6, rtol=1e-5)
    for name, reference_parameter in reference_model.named_parameters():
        sparse_parameter = dict(sparse_model.named_parameters())[name]
        assert reference_parameter.grad is not None and sparse_parameter.grad is not None
        assert torch.allclose(
            sparse_parameter.grad,
            reference_parameter.grad,
            atol=2e-4,
            rtol=2e-3,
        ), f"gradient mismatch for {name}"


def test_one_adamw_update_matches_dense_path():
    reference_model = _model()
    sparse_model = _model()
    sparse_model.load_state_dict(reference_model.state_dict())
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=3e-4)
    sparse_optimizer = torch.optim.AdamW(sparse_model.parameters(), lr=3e-4)
    idx = torch.randint(256, (2, 10))
    targets = torch.randint(256, (2, 10))

    _, reference_loss = bdh_wide_gemm_trainable_forward(
        reference_model, idx, reference_model.config.n_layer, targets
    )
    _, sparse_loss = bdh_exact_sparse_decoder_forward(
        sparse_model, idx, sparse_model.config.n_layer, targets, sparse_layout="coo"
    )
    reference_loss.backward()
    sparse_loss.backward()
    reference_optimizer.step()
    sparse_optimizer.step()

    for name, reference_parameter in reference_model.named_parameters():
        sparse_parameter = dict(sparse_model.named_parameters())[name]
        assert torch.allclose(
            sparse_parameter,
            reference_parameter,
            atol=2e-6,
            rtol=2e-5,
        ), f"AdamW update mismatch for {name}"
