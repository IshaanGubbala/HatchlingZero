"""Regression tests for the Phase 1 (plans/HatchlingZero_Reality_Plan.md)
activation-sparsity and synaptic-state-norm diagnostics added to
reference/hz0h_bdh_torch.py's bdh_stream_chunk (optional `diagnostics`
out-param) and compute_activation_and_state_diagnostics (the convenience
wrapper). Pins down: the optional param doesn't change bdh_stream_chunk's
existing behavior for callers that don't use it, the diagnostic values
are real (sparsity in [0,1], state norms positive/finite), and mode
(train/eval) is restored after the read-only diagnostic call.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_torch import (
    BDH,
    BDHConfig,
    bdh_stream_chunk,
    compute_activation_and_state_diagnostics,
    init_bdh_states,
)


def _tiny_config() -> BDHConfig:
    return BDHConfig(n_layer=3, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)


def test_diagnostics_param_does_not_change_bdh_stream_chunk_output():
    """Real regression guard: bdh_stream_chunk is used by 9+ existing H2
    tests via its exact (new_states, logits) return signature -- adding
    the optional diagnostics out-param must not change that for any
    existing caller."""
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDH(config)
    idx = torch.randint(0, config.vocab_size, (2, 12))
    states = init_bdh_states(model, 2, device=idx.device)

    states_a, logits_a = bdh_stream_chunk(model, [s.clone() for s in states], idx, start_position=0)
    states_b, logits_b = bdh_stream_chunk(model, [s.clone() for s in states], idx, start_position=0, diagnostics={})

    assert torch.equal(logits_a, logits_b)
    for a, b in zip(states_a, states_b):
        assert torch.equal(a, b)


def test_diagnostics_dict_is_filled_with_real_per_layer_values():
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDH(config)
    idx = torch.randint(0, config.vocab_size, (2, 12))
    states = init_bdh_states(model, 2, device=idx.device)

    diagnostics: dict = {}
    bdh_stream_chunk(model, states, idx, start_position=0, diagnostics=diagnostics)

    assert len(diagnostics["x_sparse_fraction_zero"]) == config.n_layer
    assert len(diagnostics["y_sparse_fraction_zero"]) == config.n_layer
    for value in diagnostics["x_sparse_fraction_zero"] + diagnostics["y_sparse_fraction_zero"]:
        assert 0.0 <= value <= 1.0


def test_compute_activation_and_state_diagnostics_real_values_and_mode_restored():
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDH(config)
    idx = torch.randint(0, config.vocab_size, (2, 12))

    model.train()
    result = compute_activation_and_state_diagnostics(model, idx)
    assert model.training, "train mode should be restored after a read-only diagnostic call"

    assert len(result["state_norms"]) == config.n_layer
    for norm in result["state_norms"]:
        assert norm > 0.0 and norm == norm  # finite, not NaN
    assert 0.0 <= result["mean_activation_sparsity"] <= 1.0

    model.eval()
    compute_activation_and_state_diagnostics(model, idx)
    assert not model.training, "eval mode should also be restored, not forced to train"


def test_relu_input_produces_nonzero_sparsity():
    """Real sanity check: a model with ReLU activations on roughly
    zero-mean random init/inputs should show real (not 0%, not 100%)
    sparsity -- guards against the diagnostic accidentally measuring
    something that's always fully dense or fully dead."""
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDH(config)
    idx = torch.randint(0, config.vocab_size, (4, 24))
    result = compute_activation_and_state_diagnostics(model, idx)
    assert 0.05 < result["mean_activation_sparsity"] < 0.95
