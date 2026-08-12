"""Regression tests for reference/hz0h_bdh_vb_torch.py (HZ-BDH-VB,
Phase 2R-B, plans/HZ Phase 2R State Redesign Plan.md). Pins down: the
streaming form (bdh_vb_stream_chunk) is numerically identical to the
parallel form (BDHVB.forward) for this NEW architecture -- same
self-consistency discipline as H2's own tests for the exact-BDH oracle,
just applied to this explicitly-different, non-upstream variant. Also
pins down the real state-byte reduction is exactly what the value-
bottleneck width implies (not approximated).
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_vb_torch import BDHVB, BDHVBConfig, bdh_vb_stream_chunk, init_bdh_vb_states


def _tiny_config(d_state: int = 8) -> BDHVBConfig:
    return BDHVBConfig(n_layer=3, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0, d_state=d_state)


def test_d_state_defaults_to_n_embd_when_unset():
    config = BDHVBConfig(n_layer=2, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)
    assert config.d_state == 64


def test_streaming_single_chunk_matches_parallel_forward_exactly():
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDHVB(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 12))

    with torch.no_grad():
        logits_full, _ = model(idx)
        states = init_bdh_vb_states(model, 2)
        _states, logits_chunk = bdh_vb_stream_chunk(model, states, idx, start_position=0)

    assert torch.allclose(logits_full, logits_chunk, atol=1e-4), f"max diff {(logits_full - logits_chunk).abs().max()}"


def test_streaming_token_by_token_matches_parallel_forward_exactly():
    config = _tiny_config()
    torch.manual_seed(1)
    model = BDHVB(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 10))

    with torch.no_grad():
        logits_full, _ = model(idx)
        states = init_bdh_vb_states(model, 2)
        chunks = []
        for t in range(idx.shape[1]):
            states, logit_t = bdh_vb_stream_chunk(model, states, idx[:, t:t + 1], start_position=t)
            chunks.append(logit_t)
        logits_token = torch.cat(chunks, dim=1)

    assert torch.allclose(logits_full, logits_token, atol=1e-4), f"max diff {(logits_full - logits_token).abs().max()}"


def test_arbitrary_chunk_boundaries_match_parallel_forward():
    config = _tiny_config()
    torch.manual_seed(2)
    model = BDHVB(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 11))
    chunk_sizes = [3, 1, 4, 2, 1]
    assert sum(chunk_sizes) == 11

    with torch.no_grad():
        logits_full, _ = model(idx)
        states = init_bdh_vb_states(model, 1)
        position = 0
        chunks = []
        for size in chunk_sizes:
            states, logit_chunk = bdh_vb_stream_chunk(model, states, idx[:, position:position + size], start_position=position)
            chunks.append(logit_chunk)
            position += size
        logits_cached = torch.cat(chunks, dim=1)

    assert torch.allclose(logits_full, logits_cached, atol=1e-4)


def test_state_bytes_scale_with_d_state_not_n_embd():
    """The entire point of this architecture: state size should scale
    with d_state, not D -- confirm the real tensor shapes reflect this."""
    config_full = _tiny_config(d_state=32)
    config_quarter = _tiny_config(d_state=8)
    model_full = BDHVB(config_full)
    model_quarter = BDHVB(config_quarter)

    states_full = init_bdh_vb_states(model_full, batch_size=1)
    states_quarter = init_bdh_vb_states(model_quarter, batch_size=1)

    full_elements = sum(s.numel() for s in states_full)
    quarter_elements = sum(s.numel() for s in states_quarter)
    assert full_elements == quarter_elements * 4


def test_generate_runs_without_error():
    config = _tiny_config()
    torch.manual_seed(3)
    model = BDHVB(config)
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    out = model.generate(prompt, max_new_tokens=4, top_k=1)
    assert out.shape == (1, 9)
