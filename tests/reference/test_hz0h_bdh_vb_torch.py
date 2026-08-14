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

from reference.hz0h_bdh_vb_torch import (
    BDHVB,
    BDHVBConfig,
    bdh_vb_stream_chunk,
    bdh_vb_stream_chunk_int8_state,
    bdh_vb_stream_chunk_int8_base_delta_state,
    init_bdh_vb_states,
    init_bdh_vb_states_int8,
    init_bdh_vb_states_int8_base_delta,
)


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


# --- Phase D1: two-level base+delta INT8 state ----------------------------

def _stream_in_chunks(model, states, idx, chunk_length, step_fn):
    chunks = []
    for start in range(0, idx.shape[1], chunk_length):
        piece = idx[:, start:start + chunk_length]
        states, logits_piece = step_fn(model, states, piece, start_position=start)
        chunks.append(logits_piece)
    return states, torch.cat(chunks, dim=1)


def test_base_delta_merging_every_chunk_matches_plain_int8_state():
    config = _tiny_config()
    torch.manual_seed(4)
    model = BDHVB(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 16))

    with torch.no_grad():
        states_int8 = init_bdh_vb_states_int8(model, 2)
        _states, logits_int8 = _stream_in_chunks(model, states_int8, idx, chunk_length=4, step_fn=bdh_vb_stream_chunk_int8_state)

        states_bd = init_bdh_vb_states_int8_base_delta(model, 2)
        step_fn = lambda m, s, c, start_position: bdh_vb_stream_chunk_int8_base_delta_state(m, s, c, start_position=start_position, merge_every_k=1)
        _states, logits_bd = _stream_in_chunks(model, states_bd, idx, chunk_length=4, step_fn=step_fn)

    assert torch.allclose(logits_int8, logits_bd, atol=1e-5), f"max diff {(logits_int8 - logits_bd).abs().max()}"


def test_base_delta_never_merging_matches_plain_unquantized_state():
    config = _tiny_config()
    torch.manual_seed(5)
    model = BDHVB(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 16))

    with torch.no_grad():
        states_plain = init_bdh_vb_states(model, 2)
        _states, logits_plain = _stream_in_chunks(model, states_plain, idx, chunk_length=4, step_fn=bdh_vb_stream_chunk)

        states_bd = init_bdh_vb_states_int8_base_delta(model, 2)
        step_fn = lambda m, s, c, start_position: bdh_vb_stream_chunk_int8_base_delta_state(m, s, c, start_position=start_position, merge_every_k=10_000)
        _states, logits_bd = _stream_in_chunks(model, states_bd, idx, chunk_length=4, step_fn=step_fn)

    assert torch.allclose(logits_plain, logits_bd, atol=1e-5), f"max diff {(logits_plain - logits_bd).abs().max()}"


def test_base_delta_intermediate_k_lies_between_the_two_extremes():
    config = _tiny_config()
    torch.manual_seed(6)
    model = BDHVB(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 32))

    with torch.no_grad():
        states_plain = init_bdh_vb_states(model, 2)
        _states, logits_plain = _stream_in_chunks(model, states_plain, idx, chunk_length=4, step_fn=bdh_vb_stream_chunk)

        states_int8 = init_bdh_vb_states_int8(model, 2)
        _states, logits_int8 = _stream_in_chunks(model, states_int8, idx, chunk_length=4, step_fn=bdh_vb_stream_chunk_int8_state)

        states_bd = init_bdh_vb_states_int8_base_delta(model, 2)
        step_fn = lambda m, s, c, start_position: bdh_vb_stream_chunk_int8_base_delta_state(m, s, c, start_position=start_position, merge_every_k=16)
        _states, logits_bd = _stream_in_chunks(model, states_bd, idx, chunk_length=4, step_fn=step_fn)

    diff_bd_vs_plain = (logits_plain - logits_bd).abs().max()
    diff_int8_vs_plain = (logits_plain - logits_int8).abs().max()
    assert diff_bd_vs_plain > 0, "merge_every_k=16 should introduce some real quantization error, not be exact"
    assert diff_bd_vs_plain < diff_int8_vs_plain, "merging less often than every chunk should drift less from the unquantized state than quantizing every chunk does"


def test_base_delta_tokens_since_merge_resets_after_a_merge():
    config = _tiny_config()
    torch.manual_seed(7)
    model = BDHVB(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 8))

    with torch.no_grad():
        states = init_bdh_vb_states_int8_base_delta(model, 1)
        states, _logits = bdh_vb_stream_chunk_int8_base_delta_state(model, states, idx, start_position=0, merge_every_k=4)

    assert all(s["tokens_since_merge"] == 0 for s in states), "a merge should have happened (8 tokens >= merge_every_k=4) and reset the counter"
    assert all(torch.equal(s["delta"], torch.zeros_like(s["delta"])) for s in states)


def test_base_delta_works_on_a_genuinely_bf16_cast_model():
    """Real bug this test exists to catch: init_bdh_vb_states_int8_base_delta
    used to hardcode delta to float32 regardless of the model's own
    dtype, contradicting the plan's own D1 spec ("delta = small BF16
    recent-update state") and causing a real RuntimeError
    ("expected m1 and m2 to have the same dtype") the first time this
    function was ever run against a genuinely bf16-cast model in
    QR @ prefix_state -- every prior real run (including on real CUDA
    hardware) happened to stay in float32 throughout (a separate,
    also-real bug in the calling scripts, which never actually cast to
    bf16 despite loading bf16-trained checkpoints), so this never
    surfaced until a script was fixed to actually request bf16."""
    config = _tiny_config()
    torch.manual_seed(8)
    model = BDHVB(config).to(dtype=torch.bfloat16)
    model.attn.freqs = model.attn.freqs.to(torch.float32)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 8))

    with torch.no_grad():
        states = init_bdh_vb_states_int8_base_delta(model, 1)
        assert all(s["delta"].dtype == torch.bfloat16 for s in states), "delta must follow the model's real working dtype, not be hardcoded to float32"
        states, logits = bdh_vb_stream_chunk_int8_base_delta_state(model, states, idx, start_position=0, merge_every_k=4)

    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits.float()).all()
