"""Basic correctness tests for the B11 torch/CUDA synthetic-backbone
comparison modules (frozen backbone, equal-param adapter, latent write
controller) -- CPU-only, run before any of this is dispatched to CUDA."""
from __future__ import annotations

import torch

from reference.hz0b_b11_equal_param_adapter_torch import EqualParamAdapter, param_count as adapter_param_count
from reference.hz0b_b11_latent_write_torch import LatentWriteController, param_count as memory_param_count
from reference.hz0b_b11_synthetic_backbone_torch import SyntheticFrozenBackbone


def test_synthetic_backbone_is_frozen():
    backbone = SyntheticFrozenBackbone(vocab_size=64, d_model=16, num_layers=2, num_heads=2, seed=0)
    assert all(not p.requires_grad for p in backbone.parameters())


def test_synthetic_backbone_output_shape_and_causality():
    backbone = SyntheticFrozenBackbone(vocab_size=64, d_model=16, num_layers=2, num_heads=2, seed=0)
    tokens = torch.randint(0, 64, (2, 5))
    hidden = backbone(tokens)
    assert hidden.shape == (2, 5, 16)
    # causal: changing a LATER token must not change an EARLIER position's hidden state
    perturbed = tokens.clone()
    perturbed[:, 4] = (perturbed[:, 4] + 1) % 64
    hidden_perturbed = backbone(perturbed)
    assert torch.allclose(hidden[:, :4, :], hidden_perturbed[:, :4, :], atol=1e-5)


def test_synthetic_backbone_deterministic_given_seed():
    a = SyntheticFrozenBackbone(vocab_size=64, d_model=16, num_layers=2, num_heads=2, seed=7)
    b = SyntheticFrozenBackbone(vocab_size=64, d_model=16, num_layers=2, num_heads=2, seed=7)
    tokens = torch.randint(0, 64, (2, 5))
    assert torch.allclose(a(tokens), b(tokens))


def test_adapter_param_count_matches_real_module():
    adapter = EqualParamAdapter(d_model=32, hidden=16, seed=0)
    counted = sum(p.numel() for p in adapter.parameters())
    assert counted == adapter_param_count(32, 16)


def test_latent_write_controller_param_count_matches_real_module():
    controller = LatentWriteController(d_model=32, key_dim=8, value_dim=8, seed=0)
    counted = sum(p.numel() for p in controller.parameters())
    assert counted == memory_param_count(32, 8, 8)


def test_latent_write_controller_forward_shapes_and_gradients_flow():
    controller = LatentWriteController(d_model=16, key_dim=4, value_dim=4, seed=0)
    hidden = torch.randn(2, 6, 16, requires_grad=False)
    output, gates = controller(hidden, num_slots=4)
    assert output.shape == (2, 6, 16)
    assert gates.shape == (2, 6)
    assert torch.all((gates >= 0) & (gates <= 1))
    loss = output.sum() + gates.sum()
    loss.backward()
    assert controller.write_gate_proj.weight.grad is not None
    assert controller.key_proj.weight.grad is not None
