"""Correctness + regression test for the Arm B efficiency measurement
(scripts/hz0h_h3t_arm_b_efficiency.py). Confirms the custom forward loop
used for fair timing exactly matches BDH.forward()'s real computation
(the thing that would make a timing comparison meaningless if it silently
diverged), and that encoder.requires_grad=False doesn't corrupt other
parameters' gradients.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig


def test_custom_forward_matches_real_bdh_forward_exactly():
    torch.manual_seed(0)
    config = BDHConfig(n_layer=3, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)
    model = BDH(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 6))

    _logits_ref, loss_ref = model(idx, targets=idx)

    B, T = idx.shape
    D, nh = config.n_embd, config.n_head
    N = D * config.mlp_internal_dim_multiplier // nh
    x = model.ln(model.embed(idx).unsqueeze(1))
    for _level in range(config.n_layer):
        x_latent = x @ model.encoder
        x_sparse = torch.relu(x_latent)
        yKV = model.ln(model.attn(Q=x_sparse, K=x_sparse, V=x))
        y_latent = yKV @ model.encoder_v
        y_sparse = torch.relu(y_latent)
        xy_sparse = model.drop(x_sparse * y_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder
        y = model.ln(yMLP)
        x = model.ln(x + y)
    logits = x.view(B, T, D) @ model.lm_head
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), idx.view(-1))

    assert abs(float(loss_ref) - float(loss)) < 1e-6


def test_encoder_requires_grad_false_does_not_corrupt_other_gradients():
    """The real lever the efficiency script relies on: setting
    encoder.requires_grad=False should skip ONLY encoder's own gradient,
    while every other parameter still gets the exact same gradient it
    would with encoder.requires_grad=True (since encoder's VALUE still
    participates in the forward computation regardless)."""
    config = BDHConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)
    torch.manual_seed(0)
    idx = torch.randint(0, config.vocab_size, (2, 6))

    # both models must be constructed from the SAME RNG state to get
    # identical initial weights -- resetting to seed 0 again (not just
    # once at the top) right before EACH construction, since idx's own
    # draw already advanced the RNG state once.
    torch.manual_seed(1)
    model_true = BDH(config)
    model_true.encoder.requires_grad_(True)
    _l, loss_true = model_true(idx, targets=idx)
    loss_true.backward()
    assert model_true.encoder.grad is not None

    torch.manual_seed(1)
    model_false = BDH(config)
    model_false.encoder.requires_grad_(False)
    _l2, loss_false = model_false(idx, targets=idx)
    loss_false.backward()
    assert model_false.encoder.grad is None

    assert abs(float(loss_true) - float(loss_false)) < 1e-6
    assert torch.allclose(model_true.encoder_v.grad, model_false.encoder_v.grad, atol=1e-6)
    assert torch.allclose(model_true.decoder.grad, model_false.decoder.grad, atol=1e-6)
    assert torch.allclose(model_true.lm_head.grad, model_false.lm_head.grad, atol=1e-6)
