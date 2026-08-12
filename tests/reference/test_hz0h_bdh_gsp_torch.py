"""Regression tests for reference/hz0h_bdh_gs_torch.py's BDHGSP
(Grouped Synaptic State with per-layer Projections, Phase 2R-C v2,
plans/HZ Phase 2R State Redesign Plan.md). Pins down a real design bug
that was caught and fixed before any training happened: an earlier
version of `forward()` computed each layer's attention independently
over the whole sequence in one shot, which NEVER exercises cross-layer
group state sharing at all (state only accumulates across separate
streaming calls) -- training that way would teach the model a
computation it would never run at real (multi-call, grouped-streaming)
inference time. Fixed by making `forward()` itself run the identical
token-by-token mechanism `bdh_gsp_stream_chunk` uses at inference.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_gs_torch import BDHGSP, BDHGSPConfig, bdh_gsp_stream_chunk, init_bdh_gsp_states


def _tiny_config(n_state_groups: int = 2) -> BDHGSPConfig:
    return BDHGSPConfig(n_layer=6, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0, n_state_groups=n_state_groups)


def test_n_state_groups_defaults_to_n_layer_when_unset():
    config = BDHGSPConfig(n_layer=4, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)
    assert config.n_state_groups == 4


def test_forward_matches_manual_token_by_token_streaming_exactly():
    """forward() IS the token-by-token loop (not a separate
    reimplementation) -- this should be exact by construction, and this
    test pins that down so a future refactor can't silently reintroduce
    the train/inference mismatch this design was built to avoid."""
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDHGSP(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 12))

    with torch.no_grad():
        logits_forward, _ = model(idx)

        states = init_bdh_gsp_states(model, 2)
        chunks = []
        for t in range(12):
            states, logit_t = bdh_gsp_stream_chunk(model, states, idx[:, t:t + 1], start_position=t)
            chunks.append(logit_t)
        logits_manual = torch.cat(chunks, dim=1)

    assert torch.equal(logits_forward, logits_manual)


def test_single_large_chunk_genuinely_differs_from_token_by_token():
    """Real, DISCLOSED architectural property (unlike exact BDH, which
    H2 proved is exactly chunk-boundary-invariant): this grouped-state
    design is NOT chunk-invariant -- cross-layer sharing only manifests
    across separate streaming calls, so a single big chunk (no
    intermediate calls) exercises LESS cross-layer sharing than
    token-by-token decode. This is expected, not a bug -- pinned down so
    it's never mistaken for one."""
    config = _tiny_config()
    torch.manual_seed(1)
    model = BDHGSP(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 10))

    with torch.no_grad():
        states_a = init_bdh_gsp_states(model, 1)
        _states_a, logits_single_chunk = bdh_gsp_stream_chunk(model, states_a, idx, start_position=0)

        states_b = init_bdh_gsp_states(model, 1)
        chunks = []
        for t in range(10):
            states_b, logit_t = bdh_gsp_stream_chunk(model, states_b, idx[:, t:t + 1], start_position=t)
            chunks.append(logit_t)
        logits_token_by_token = torch.cat(chunks, dim=1)

    assert not torch.allclose(logits_single_chunk, logits_token_by_token, atol=1e-3)


def test_no_sharing_case_n_state_groups_equals_n_layer_still_differs_from_exact_bdh():
    """Even with n_state_groups == n_layer (no state SHARING at all),
    BDHGSP is not exact BDH -- it has real, extra P_l/O_l projections
    that change the computation. Confirms this class is honestly a
    different architecture, not silently falling back to exact BDH's
    own numbers when grouping is disabled."""
    from reference.hz0h_bdh_torch import BDH, BDHConfig

    torch.manual_seed(2)
    bdh_config = BDHConfig(n_layer=4, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)
    exact_model = BDH(bdh_config)
    exact_model.eval()

    torch.manual_seed(2)
    gsp_config = BDHGSPConfig(n_layer=4, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0, n_state_groups=4)
    gsp_model = BDHGSP(gsp_config)
    gsp_model.eval()

    idx = torch.randint(0, 32, (1, 6))
    with torch.no_grad():
        exact_logits, _ = exact_model(idx)
        gsp_logits, _ = gsp_model(idx)

    assert not torch.allclose(exact_logits, gsp_logits, atol=1e-2)


def test_gradients_flow_through_the_sequential_streaming_loop():
    config = _tiny_config()
    torch.manual_seed(3)
    model = BDHGSP(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (2, 8))
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
    _logits, loss = model(x, targets=y)
    loss.backward()

    assert model.P[0].grad is not None and float(model.P[0].grad.norm()) > 0
    assert model.O[0].grad is not None and float(model.O[0].grad.norm()) > 0
    assert model.encoder.grad is not None and float(model.encoder.grad.norm()) > 0
