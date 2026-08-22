from __future__ import annotations

import torch

from reference.hz0h_bdh_symmetric_backward_broadcast_torch import (
    bdh_symmetric_broadcast_backward_attention,
    bdh_symmetric_broadcast_backward_forward,
)
from reference.hz0h_bdh_torch import Attention, BDH, BDHConfig
from reference.hz0h_bdh_wide_gemm_trainable_torch import bdh_wide_gemm_trainable_forward


def _model(seed: int = 23) -> BDH:
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


def test_symmetric_broadcast_attention_matches_output_and_input_gradients():
    torch.manual_seed(5)
    attention = Attention(BDHConfig(n_embd=24, n_head=4, mlp_internal_dim_multiplier=8))
    q_a = torch.randn(2, 4, 9, 48, requires_grad=True)
    v_a = torch.randn(2, 1, 9, 24, requires_grad=True)
    q_b = q_a.detach().clone().requires_grad_(True)
    v_b = v_a.detach().clone().requires_grad_(True)
    gradient = torch.randn(2, 4, 9, 24)

    reference = attention(q_a, q_a, v_a)
    candidate = bdh_symmetric_broadcast_backward_attention(q_b, v_b, attention.freqs)
    reference.backward(gradient)
    candidate.backward(gradient)

    assert torch.allclose(candidate, reference, atol=1e-5, rtol=1e-5)
    assert torch.allclose(q_b.grad, q_a.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(v_b.grad, v_a.grad, atol=2e-5, rtol=2e-5)


def test_full_model_matches_all_parameter_gradients():
    reference_model = _model()
    candidate_model = _model()
    candidate_model.load_state_dict(reference_model.state_dict())
    idx = torch.randint(256, (2, 10))
    targets = torch.randint(256, (2, 10))

    reference_logits, reference_loss = bdh_wide_gemm_trainable_forward(
        reference_model, idx, reference_model.config.n_layer, targets
    )
    candidate_logits, candidate_loss = bdh_symmetric_broadcast_backward_forward(
        candidate_model, idx, candidate_model.config.n_layer, targets
    )
    reference_loss.backward()
    candidate_loss.backward()

    assert torch.allclose(candidate_logits, reference_logits, atol=1e-5, rtol=1e-4)
    assert torch.allclose(candidate_loss, reference_loss, atol=1e-6, rtol=1e-5)
    candidate_parameters = dict(candidate_model.named_parameters())
    for name, reference_parameter in reference_model.named_parameters():
        candidate_parameter = candidate_parameters[name]
        assert reference_parameter.grad is not None and candidate_parameter.grad is not None
        assert torch.allclose(
            candidate_parameter.grad,
            reference_parameter.grad,
            atol=2e-4,
            rtol=2e-3,
        ), f"gradient mismatch for {name}"


def test_one_adamw_update_matches_reference():
    reference_model = _model()
    candidate_model = _model()
    candidate_model.load_state_dict(reference_model.state_dict())
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=3e-4)
    candidate_optimizer = torch.optim.AdamW(candidate_model.parameters(), lr=3e-4)
    idx = torch.randint(256, (2, 10))
    targets = torch.randint(256, (2, 10))

    _, reference_loss = bdh_wide_gemm_trainable_forward(
        reference_model, idx, reference_model.config.n_layer, targets
    )
    _, candidate_loss = bdh_symmetric_broadcast_backward_forward(
        candidate_model, idx, candidate_model.config.n_layer, targets
    )
    reference_loss.backward()
    candidate_loss.backward()
    reference_optimizer.step()
    candidate_optimizer.step()

    for name, reference_parameter in reference_model.named_parameters():
        candidate_parameter = dict(candidate_model.named_parameters())[name]
        assert torch.allclose(
            candidate_parameter,
            reference_parameter,
            atol=2e-6,
            rtol=2e-5,
        ), f"AdamW update mismatch for {name}"


def test_matches_original_symmetric_backward():
    """The broadcast rewrite must match d83db47's version, not just the
    oracle -- this isolates that the rewrite changed only how value's
    broadcast is realized, nothing numerical. Not bit-exact: matmul's
    broadcast reduction and the original's explicit bmm+sum(dim=1) sum
    heads in a different order, so float32 rounding differs at the
    ~1e-6-vs-magnitude-~10 level (verified negligible before relaxing this
    from an exact-equality check)."""
    from reference.hz0h_bdh_symmetric_backward_torch import bdh_symmetric_backward_attention

    torch.manual_seed(7)
    attention = Attention(BDHConfig(n_embd=24, n_head=4, mlp_internal_dim_multiplier=8))
    q_a = torch.randn(2, 4, 9, 48, requires_grad=True)
    v_a = torch.randn(2, 1, 9, 24, requires_grad=True)
    q_b = q_a.detach().clone().requires_grad_(True)
    v_b = v_a.detach().clone().requires_grad_(True)
    gradient = torch.randn(2, 4, 9, 24)

    original = bdh_symmetric_backward_attention(q_a, v_a, attention.freqs)
    broadcast = bdh_symmetric_broadcast_backward_attention(q_b, v_b, attention.freqs)
    original.backward(gradient)
    broadcast.backward(gradient)

    assert torch.allclose(broadcast, original, atol=1e-5, rtol=1e-5)
    assert torch.allclose(q_b.grad, q_a.grad, atol=1e-4, rtol=1e-5)
    assert torch.allclose(v_b.grad, v_a.grad, atol=1e-4, rtol=1e-5)
