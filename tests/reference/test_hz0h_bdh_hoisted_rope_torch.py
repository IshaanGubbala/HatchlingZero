from __future__ import annotations

import torch

from reference.hz0h_bdh_hoisted_rope_torch import bdh_hoisted_rope_forward
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _model(seed: int = 7) -> BDH:
    torch.manual_seed(seed)
    return BDH(
        BDHConfig(
            n_layer=3,
            n_embd=32,
            n_head=4,
            mlp_internal_dim_multiplier=8,
            vocab_size=64,
            dropout=0.0,
        )
    )


def test_hoisted_rope_matches_oracle_logits_and_loss_exactly():
    model = _model().eval()
    idx = torch.randint(0, 64, (2, 17))
    targets = torch.randint(0, 64, (2, 17))
    with torch.no_grad():
        oracle_logits, oracle_loss = model(idx, targets)
        hoisted_logits, hoisted_loss = bdh_hoisted_rope_forward(
            model, idx, targets
        )
    assert torch.equal(oracle_logits, hoisted_logits)
    assert torch.equal(oracle_loss, hoisted_loss)


def test_hoisted_rope_matches_every_named_parameter_gradient():
    oracle = _model()
    hoisted = _model()
    idx = torch.randint(0, 64, (2, 13))
    targets = torch.randint(0, 64, (2, 13))

    _, oracle_loss = oracle(idx, targets)
    oracle_loss.backward()
    _, hoisted_loss = bdh_hoisted_rope_forward(hoisted, idx, targets)
    hoisted_loss.backward()

    oracle_parameters = dict(oracle.named_parameters())
    for name, parameter in hoisted.named_parameters():
        expected = oracle_parameters[name].grad
        assert expected is not None
        assert parameter.grad is not None
        assert torch.equal(expected, parameter.grad), name


def test_hoisted_rope_is_one_compiler_graph_without_breaks():
    model = _model()
    idx = torch.randint(0, 64, (1, 8))
    targets = torch.randint(0, 64, (1, 8))

    explanation = torch._dynamo.explain(bdh_hoisted_rope_forward)(
        model, idx, targets
    )
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0
