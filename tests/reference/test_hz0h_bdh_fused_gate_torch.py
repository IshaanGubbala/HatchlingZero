"""Real correctness tests for reference/hz0h_bdh_fused_gate_torch.py and
reference/hz0h_bdh_packed_encoder_fused_gate_torch.py. Requires CUDA
(Triton) -- these are skipped, not failed, when no GPU is available so the
suite stays runnable on CPU-only machines."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused gate kernel requires CUDA/Triton")


def test_fused_gate_matches_relu_then_multiply_forward():
    from reference.hz0h_bdh_fused_gate_torch import fused_gate

    torch.manual_seed(3)
    x_sparse = torch.randn(4, 8, 16, 32, device="cuda")
    y_latent = torch.randn(4, 8, 16, 32, device="cuda")
    expected = x_sparse * F.relu(y_latent)
    actual = fused_gate(x_sparse, y_latent)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_fused_gate_matches_relu_then_multiply_backward():
    from reference.hz0h_bdh_fused_gate_torch import fused_gate

    torch.manual_seed(5)
    x_a = torch.randn(4, 8, 16, 32, device="cuda", requires_grad=True)
    y_a = torch.randn(4, 8, 16, 32, device="cuda", requires_grad=True)
    x_b = x_a.detach().clone().requires_grad_(True)
    y_b = y_a.detach().clone().requires_grad_(True)
    grad = torch.randn(4, 8, 16, 32, device="cuda")

    reference = x_a * F.relu(y_a)
    candidate = fused_gate(x_b, y_b)
    reference.backward(grad)
    candidate.backward(grad)

    assert torch.allclose(candidate, reference, atol=1e-6, rtol=1e-6)
    assert torch.allclose(x_b.grad, x_a.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(y_b.grad, y_a.grad, atol=1e-6, rtol=1e-6)


def test_fused_gate_zero_at_relu_boundary_gets_zero_gradient():
    """y_latent exactly 0 must behave like relu's own boundary (zero
    gradient contribution from that entry), not NaN or a stray value."""
    from reference.hz0h_bdh_fused_gate_torch import fused_gate

    x = torch.tensor([1.0, 2.0, -3.0], device="cuda", requires_grad=True)
    y = torch.tensor([0.0, -1.0, 5.0], device="cuda", requires_grad=True)
    out = fused_gate(x, y)
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()
    assert out[1].item() == 0.0  # relu(-1) = 0
    assert y.grad[1].item() == 0.0  # relu'(-1) = 0


def test_full_model_matches_unfused_packed_encoder():
    from reference.hz0h_bdh_packed_encoder_torch import PackedEncoderBDH, bdh_packed_encoder_forward_checkpointed
    from reference.hz0h_bdh_packed_encoder_fused_gate_torch import bdh_packed_encoder_fused_gate_forward_checkpointed
    from reference.hz0h_bdh_torch import BDHConfig

    config = BDHConfig(n_layer=5, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=256, dropout=0.0)
    torch.manual_seed(13)
    unfused_model = PackedEncoderBDH(config).cuda()
    torch.manual_seed(13)
    fused_model = PackedEncoderBDH(config).cuda()

    idx = torch.randint(256, (2, 12), device="cuda")
    targets = torch.randint(256, (2, 12), device="cuda")

    unfused_logits, unfused_loss = bdh_packed_encoder_forward_checkpointed(
        unfused_model, idx, unfused_model.config.n_layer, targets, checkpoint_segment_size=1,
    )
    unfused_loss.backward()

    fused_logits, fused_loss = bdh_packed_encoder_fused_gate_forward_checkpointed(
        fused_model, idx, fused_model.config.n_layer, targets, checkpoint_segment_size=1,
    )
    fused_loss.backward()

    assert torch.allclose(unfused_logits, fused_logits, atol=1e-5, rtol=1e-4)
    assert torch.allclose(unfused_loss, fused_loss, atol=1e-5, rtol=1e-4)
    unfused_params = dict(unfused_model.named_parameters())
    fused_params = dict(fused_model.named_parameters())
    for name in unfused_params:
        ga, gb = unfused_params[name].grad, fused_params[name].grad
        assert ga is not None and gb is not None, f"{name} missing a gradient"
        assert torch.allclose(ga, gb, atol=1e-4, rtol=1e-3), f"gradient mismatch at {name}"


def test_one_adamw_step_matches_unfused_packed_encoder():
    from reference.hz0h_bdh_packed_encoder_torch import PackedEncoderBDH, bdh_packed_encoder_forward_checkpointed
    from reference.hz0h_bdh_packed_encoder_fused_gate_torch import bdh_packed_encoder_fused_gate_forward_checkpointed
    from reference.hz0h_bdh_torch import BDHConfig

    config = BDHConfig(n_layer=5, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=256, dropout=0.0)
    torch.manual_seed(19)
    unfused_model = PackedEncoderBDH(config).cuda()
    torch.manual_seed(19)
    fused_model = PackedEncoderBDH(config).cuda()
    unfused_opt = torch.optim.AdamW(unfused_model.parameters(), lr=3e-4)
    fused_opt = torch.optim.AdamW(fused_model.parameters(), lr=3e-4)

    idx = torch.randint(256, (2, 12), device="cuda")
    targets = torch.randint(256, (2, 12), device="cuda")

    _, unfused_loss = bdh_packed_encoder_forward_checkpointed(unfused_model, idx, unfused_model.config.n_layer, targets)
    unfused_loss.backward()
    unfused_opt.step()

    _, fused_loss = bdh_packed_encoder_fused_gate_forward_checkpointed(fused_model, idx, fused_model.config.n_layer, targets)
    fused_loss.backward()
    fused_opt.step()

    unfused_params = dict(unfused_model.named_parameters())
    fused_params = dict(fused_model.named_parameters())
    for name in unfused_params:
        assert torch.allclose(unfused_params[name], fused_params[name], atol=1e-4, rtol=1e-3), (
            f"post-AdamW mismatch at {name}"
        )
