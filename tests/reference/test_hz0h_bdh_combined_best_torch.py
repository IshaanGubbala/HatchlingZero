"""Real correctness tests for reference/hz0h_bdh_combined_best_torch.py.

Load-bearing gate: at num_jumps=0, combined_bdh_forward must reproduce
ablated_bdh_forward(..., use_softmax=True, scale_scores=True) EXACTLY
-- the softmax_scaled attention here must be the SAME tested formula
from Part 3, not a fresh reimplementation."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_combined_best_torch import combined_bdh_forward, combined_bdh_forward_with_trajectory
from reference.hz0h_bdh_jump_operator_torch import JumpOperator
from reference.hz0h_bdh_primitive_ablations_torch import ablated_bdh_forward
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _model(seed: int = 7, **overrides) -> BDH:
    torch.manual_seed(seed)
    config = BDHConfig(
        n_layer=overrides.get("n_layer", 6), n_embd=overrides.get("n_embd", 32),
        n_head=overrides.get("n_head", 4), mlp_internal_dim_multiplier=overrides.get("mult", 8),
        vocab_size=256, dropout=0.0,
    )
    return BDH(config).eval()


def test_zero_jumps_matches_softmax_scaled_ablation_exactly():
    model = _model()
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    with torch.no_grad():
        reference_logits, reference_loss = ablated_bdh_forward(
            model, idx, model.config.n_layer, targets, use_softmax=True, scale_scores=True,
        )
        combined_logits, combined_loss = combined_bdh_forward(
            model, None, idx, real_prefix_iterations=model.config.n_layer, num_jumps=0, targets=targets,
        )
    assert torch.equal(reference_logits, combined_logits)
    assert torch.equal(reference_loss, combined_loss)


def test_trajectory_final_state_matches_forward_final_logits():
    model = _model(n_layer=5)
    idx = torch.randint(256, (2, 12))
    D = model.config.n_embd
    B, T = idx.shape
    with torch.no_grad():
        x_states = combined_bdh_forward_with_trajectory(model, idx, 5)
        logits, _ = combined_bdh_forward(model, None, idx, real_prefix_iterations=5, num_jumps=0)
        derived_logits = x_states[-1].view(B, T, D) @ model.lm_head
    assert torch.equal(derived_logits, logits)
    assert len(x_states) == 6


def test_zero_init_jump_is_a_pure_noop_on_the_combined_recipe():
    model = _model(n_layer=6)
    jump = JumpOperator(d_model=model.config.n_embd, hidden_mult=4, jump_size=2)
    idx = torch.randint(256, (2, 12))
    with torch.no_grad():
        no_jumps, _ = combined_bdh_forward(model, jump, idx, real_prefix_iterations=2, num_jumps=0)
        with_jumps, _ = combined_bdh_forward(model, jump, idx, real_prefix_iterations=2, num_jumps=3)
    assert torch.equal(no_jumps, with_jumps)
