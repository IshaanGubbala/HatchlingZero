"""Real correctness tests for reference/hz0h_bdh_trajectory_torch.py.

Load-bearing gate: at depth=model.config.n_layer, `bdh_forward_with_trajectory`
must reproduce the real oracle's logits/loss EXACTLY -- every diagnostic
this module can support is meaningless if the captured trajectory isn't
genuinely BDH's own recurrence."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_trajectory_torch import bdh_forward_with_trajectory


def _model(seed: int = 7, **overrides) -> BDH:
    torch.manual_seed(seed)
    config = BDHConfig(
        n_layer=overrides.get("n_layer", 5), n_embd=overrides.get("n_embd", 32),
        n_head=overrides.get("n_head", 4), mlp_internal_dim_multiplier=overrides.get("mult", 8),
        vocab_size=256, dropout=0.0,
    )
    return BDH(config).eval()


def test_trajectory_forward_matches_the_real_oracle_exactly_at_full_depth():
    model = _model()
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    with torch.no_grad():
        oracle_logits, oracle_loss = model(idx, targets)
        traj_logits, traj_loss, x_states, relu_masks = bdh_forward_with_trajectory(
            model, idx, model.config.n_layer, targets,
        )
    assert torch.equal(oracle_logits, traj_logits), "trajectory forward is not the real oracle"
    assert torch.equal(oracle_loss, traj_loss)


def test_trajectory_lengths_and_shapes_are_correct():
    model = _model(n_layer=6)
    idx = torch.randint(256, (2, 12))
    depth = 6
    with torch.no_grad():
        _, _, x_states, relu_masks = bdh_forward_with_trajectory(model, idx, depth)
    assert len(x_states) == depth + 1, "x_states must include the pre-recurrence state x_0"
    assert len(relu_masks) == depth
    B, T, D = 2, 12, model.config.n_embd
    for state in x_states:
        assert state.shape == (B, 1, T, D)
    nh = model.config.n_head
    N = D * model.config.mlp_internal_dim_multiplier // nh
    for mask in relu_masks:
        assert mask.shape == (B, nh, T, N)
        assert mask.dtype == torch.bool


def test_trajectory_states_actually_change_across_iterations():
    """Real sanity check: successive x_r must genuinely differ (the
    recurrence is doing something), not be a frozen/no-op loop."""
    model = _model(n_layer=5)
    idx = torch.randint(256, (2, 12))
    with torch.no_grad():
        _, _, x_states, _ = bdh_forward_with_trajectory(model, idx, 5)
    for r in range(len(x_states) - 1):
        assert not torch.allclose(x_states[r], x_states[r + 1]), (
            f"state at step {r} is identical to step {r + 1} -- recurrence is a no-op"
        )


def test_variable_depth_matches_the_variable_depth_reference():
    """At depth < n_layer, the trajectory forward's FINAL state must
    match `bdh_variable_depth_forward`'s own logits exactly -- confirms
    trajectory capture doesn't change the underlying computation."""
    from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward
    model = _model(n_layer=8)
    idx = torch.randint(256, (2, 12))
    with torch.no_grad():
        reference_logits, _ = bdh_variable_depth_forward(model, idx, 3)
        traj_logits, _, x_states, _ = bdh_forward_with_trajectory(model, idx, 3)
    assert torch.equal(reference_logits, traj_logits)
    assert len(x_states) == 4
