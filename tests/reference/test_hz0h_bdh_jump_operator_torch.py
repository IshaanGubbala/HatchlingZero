"""Real correctness tests for reference/hz0h_bdh_jump_operator_torch.py.

Load-bearing gate: at num_jumps=0, jump_bdh_forward must reproduce
bdh_variable_depth_forward's own output exactly (the jump operator is
never invoked in that case, so this checks the real-iteration path
alone is correct)."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_jump_operator_torch import JumpOperator, jump_bdh_forward
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward


def _model(seed: int = 7, **overrides) -> BDH:
    torch.manual_seed(seed)
    config = BDHConfig(
        n_layer=overrides.get("n_layer", 6), n_embd=overrides.get("n_embd", 32),
        n_head=overrides.get("n_head", 4), mlp_internal_dim_multiplier=overrides.get("mult", 8),
        vocab_size=256, dropout=0.0,
    )
    return BDH(config).eval()


def test_zero_jumps_matches_bdh_variable_depth_forward_exactly():
    model = _model()
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    with torch.no_grad():
        reference_logits, reference_loss = bdh_variable_depth_forward(model, idx, 4, targets)
        jump_logits, jump_loss = jump_bdh_forward(model, None, idx, real_prefix_iterations=4, num_jumps=0, targets=targets)
    assert torch.equal(reference_logits, jump_logits)
    assert torch.equal(reference_loss, jump_loss)


def test_jump_operator_zero_init_is_identity():
    """JumpOperator's final layer is zero-initialized, so at
    construction J(x) == x exactly -- a clean starting point for
    distillation training (predicting 'do nothing extra' by default)."""
    jump = JumpOperator(d_model=32, hidden_mult=4, jump_size=2)
    x = torch.randn(2, 1, 12, 32)
    with torch.no_grad():
        out = jump(x)
    assert torch.equal(out, x)


def test_jumps_change_output_after_training_step():
    model = _model()
    jump = JumpOperator(d_model=model.config.n_embd, hidden_mult=4, jump_size=2)
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))

    with torch.no_grad():
        zero_init_logits, _ = jump_bdh_forward(model, jump, idx, real_prefix_iterations=2, num_jumps=1, targets=targets)
        reference_logits, _ = bdh_variable_depth_forward(model, idx, 2, targets)
    assert torch.equal(zero_init_logits, reference_logits), (
        "at zero-init, a jump is a pure no-op, so real_prefix=2 + 1 jump must equal real depth=2 exactly"
    )

    optimizer = torch.optim.SGD(jump.parameters(), lr=1.0)
    _, loss = jump_bdh_forward(model, jump, idx, real_prefix_iterations=2, num_jumps=1, targets=targets)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        trained_logits, _ = jump_bdh_forward(model, jump, idx, real_prefix_iterations=2, num_jumps=1, targets=targets)
    assert not torch.allclose(trained_logits, zero_init_logits), "a real gradient step must change the jump's output"


def test_zero_init_jumps_are_a_pure_noop_regardless_of_count():
    """Real structural check: at zero-init, ANY number of jumps must be
    exactly equivalent to zero jumps -- jump() is pure identity until
    trained, so 4 jumps of a zero-init operator changes nothing."""
    model = _model(n_layer=8)
    jump = JumpOperator(d_model=model.config.n_embd, hidden_mult=4, jump_size=2)
    idx = torch.randint(256, (2, 12))
    with torch.no_grad():
        no_jumps, _ = jump_bdh_forward(model, jump, idx, real_prefix_iterations=2, num_jumps=0)
        four_jumps, _ = jump_bdh_forward(model, jump, idx, real_prefix_iterations=2, num_jumps=4)
    assert torch.equal(no_jumps, four_jumps)
