"""Regression tests for reference/hz0h_bdh_soft_grouped_state_torch.py's
BDHSoftGroupedState (HZ Phase 2R Step 2, plans/HZ Integrated Candidate
Plan.md) -- the one authorized redesign attempt for grouped state,
learned soft addressing over k shared banks instead of BDHGSP's hard
depth-block assignment + per-layer projections. Same discipline as
test_hz0h_bdh_gsp_torch.py: forward() must BE the token-by-token
streaming loop (not a separate reimplementation), since a shared bank
only accumulates across separate streaming calls.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_soft_grouped_state_torch import (
    BDHSoftGroupedConfig,
    BDHSoftGroupedState,
    bdh_soft_grouped_stream_chunk,
    init_bdh_soft_grouped_states,
)


def _tiny_config(n_state_banks: int = 2) -> BDHSoftGroupedConfig:
    return BDHSoftGroupedConfig(n_layer=6, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0, n_state_banks=n_state_banks)


def test_n_state_banks_defaults_to_n_layer_when_unset():
    config = BDHSoftGroupedConfig(n_layer=4, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)
    assert config.n_state_banks == 4


def test_address_logits_shape_and_softmax_sums_to_one():
    config = _tiny_config(n_state_banks=3)
    torch.manual_seed(0)
    model = BDHSoftGroupedState(config)
    assert model.address_logits.shape == (6, 3)
    p = F.softmax(model.address_logits, dim=-1)
    assert torch.allclose(p.sum(dim=-1), torch.ones(6), atol=1e-6)


def test_forward_matches_manual_token_by_token_streaming_exactly():
    """forward() IS the token-by-token loop -- pins that down so a
    future refactor can't silently reintroduce the train/inference
    mismatch BDHGSP's own tests already caught once for the hard-
    grouping design."""
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDHSoftGroupedState(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 12))

    with torch.no_grad():
        logits_forward, _ = model(idx)

        states = init_bdh_soft_grouped_states(model, 2)
        chunks = []
        for t in range(12):
            states, logit_t = bdh_soft_grouped_stream_chunk(model, states, idx[:, t:t + 1], start_position=t)
            chunks.append(logit_t)
        logits_manual = torch.cat(chunks, dim=1)

    assert torch.equal(logits_forward, logits_manual)


def test_gradients_flow_through_address_logits_and_shared_weights():
    config = _tiny_config()
    torch.manual_seed(1)
    model = BDHSoftGroupedState(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (2, 8))
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
    _logits, loss = model(x, targets=y)
    loss.backward()

    assert model.address_logits.grad is not None and float(model.address_logits.grad.norm()) > 0
    assert model.encoder.grad is not None and float(model.encoder.grad.norm()) > 0


def test_single_large_chunk_differs_from_token_by_token():
    """Same real, disclosed non-chunk-invariance as BDHGSP: cross-layer
    bank sharing only manifests across separate streaming calls."""
    config = _tiny_config()
    torch.manual_seed(2)
    model = BDHSoftGroupedState(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 10))

    with torch.no_grad():
        states_a = init_bdh_soft_grouped_states(model, 1)
        _states_a, logits_single_chunk = bdh_soft_grouped_stream_chunk(model, states_a, idx, start_position=0)

        states_b = init_bdh_soft_grouped_states(model, 1)
        chunks = []
        for t in range(10):
            states_b, logit_t = bdh_soft_grouped_stream_chunk(model, states_b, idx[:, t:t + 1], start_position=t)
            chunks.append(logit_t)
        logits_token_by_token = torch.cat(chunks, dim=1)

    assert not torch.allclose(logits_single_chunk, logits_token_by_token, atol=1e-3)


def test_n_state_banks_equals_one_still_runs_and_pools_every_layer():
    """Real edge case: k=1 (maximum compression) should still run --
    every layer's soft address necessarily degenerates to weight 1.0 on
    the single bank."""
    config = _tiny_config(n_state_banks=1)
    torch.manual_seed(3)
    model = BDHSoftGroupedState(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 6))
    with torch.no_grad():
        logits, _ = model(idx)
    assert logits.shape == (1, 6, 32)
