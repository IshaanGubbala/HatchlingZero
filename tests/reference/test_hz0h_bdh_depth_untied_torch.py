"""Real correctness tests for reference/hz0h_bdh_depth_untied_torch.py.

Load-bearing gate: with every level's weights forced identical (via
`tie_all_levels_to`), `DepthUntiedBDH.forward` must reproduce the real
oracle `BDH.forward` EXACTLY. Every untied-vs-tied quality delta this
module can produce is meaningless if the tied special case isn't
genuinely BDH, so that equivalence is proven rather than assumed."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_depth_untied_torch import DepthUntiedBDH, budget_matched_multiplier
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _config(**overrides) -> BDHConfig:
    return BDHConfig(
        n_layer=overrides.get("n_layer", 3), n_embd=overrides.get("n_embd", 32),
        n_head=overrides.get("n_head", 4), mlp_internal_dim_multiplier=overrides.get("mult", 8),
        vocab_size=256, dropout=0.0,
    )


def test_tied_levels_reproduce_the_real_oracle_exactly():
    config = _config()
    torch.manual_seed(7)
    oracle = BDH(config).eval()
    torch.manual_seed(11)  # different seed on purpose -- init draw must not matter once tied
    untied = DepthUntiedBDH(config, depth=config.n_layer).eval()
    untied.tie_all_levels_to(oracle.encoder, oracle.encoder_v, oracle.decoder)
    untied.embed.load_state_dict(oracle.embed.state_dict())
    untied.lm_head.data.copy_(oracle.lm_head.data)

    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    with torch.no_grad():
        oracle_logits, oracle_loss = oracle(idx, targets)
        untied_logits, untied_loss = untied(idx, targets)
    assert torch.equal(oracle_logits, untied_logits), "tied-levels forward is not the real oracle"
    assert torch.equal(oracle_loss, untied_loss)


def test_genuinely_untied_levels_change_the_output():
    config = _config()
    torch.manual_seed(7)
    model = DepthUntiedBDH(config, depth=config.n_layer).eval()
    idx = torch.randint(256, (2, 12))
    with torch.no_grad():
        untied_logits, _ = model(idx)
        first_encoder = model.level_encoders[0].clone()
        model.tie_all_levels_to(first_encoder, model.level_encoders_v[0], model.level_decoders[0])
        tied_logits, _ = model(idx)
    assert torch.isfinite(untied_logits).all()
    assert not torch.allclose(untied_logits, tied_logits), "untying must change the real output"


def test_untied_gradients_are_independent_per_level():
    """Real structural check: each level's own weights must receive their
    own gradient, not a summed/shared one -- proving they are genuinely
    separate parameters, not the same tensor viewed `depth` times."""
    config = _config()
    torch.manual_seed(7)
    model = DepthUntiedBDH(config, depth=config.n_layer)
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    _, loss = model(idx, targets)
    loss.backward()
    grads = [p.grad.clone() for p in model.level_encoders]
    for i in range(len(grads)):
        for j in range(i + 1, len(grads)):
            assert not torch.equal(grads[i], grads[j]), (
                f"level {i} and level {j} encoder gradients are identical -- levels are not independent"
            )


def test_budget_matched_multiplier_divides_and_floors():
    assert budget_matched_multiplier(16, depth=4) == 4
    assert budget_matched_multiplier(16, depth=3) == 5
    assert budget_matched_multiplier(2, depth=8) == 1  # floors at 1, never 0
