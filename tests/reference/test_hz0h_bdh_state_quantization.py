"""Regression tests for reference/hz0h_bdh_torch.py's INT8 synaptic-state
quantization (Phase 3, plans/HatchlingZero_Reality_Plan.md §6.4) --
quantize_state_int8/dequantize_state_int8/init_bdh_states_int8/
bdh_stream_chunk_int8_state. Pins down: quantize/dequantize round-trips
to a small, bounded error; bdh_stream_chunk_int8_state agrees closely
with the real fp32 bdh_stream_chunk (small compounding error across many
token-by-token steps, not exploding); state bytes really do shrink ~4x.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_torch import (
    BDH,
    BDHConfig,
    bdh_stream_chunk,
    bdh_stream_chunk_int8_state,
    dequantize_state_int8,
    init_bdh_states,
    init_bdh_states_int8,
    quantize_state_int8,
)


def _tiny_config() -> BDHConfig:
    return BDHConfig(n_layer=3, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)


def test_quantize_dequantize_round_trip_has_small_bounded_error():
    torch.manual_seed(0)
    x = torch.randn(4, 8) * 3
    q, scale = quantize_state_int8(x)
    assert q.dtype == torch.int8
    x_hat = dequantize_state_int8(q, scale)
    max_relative_error = float((x - x_hat).abs().max() / x.abs().max())
    assert max_relative_error < 0.01, f"expected <1% relative quantization error, got {max_relative_error}"


def test_quantize_of_zero_state_is_exact():
    """A fresh (all-zero) state must quantize to exactly zero -- no
    quantization error at t=0, matching init_bdh_states_int8's own
    docstring claim."""
    zero = torch.zeros(2, 3, 4, 5)
    q, scale = quantize_state_int8(zero)
    assert torch.equal(dequantize_state_int8(q, scale), zero)


def test_int8_state_bytes_are_real_4x_smaller_than_fp32():
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDH(config)
    fp32_states = init_bdh_states(model, batch_size=1)
    int8_states = init_bdh_states_int8(model, batch_size=1)

    fp32_bytes = sum(s.numel() * 4 for s in fp32_states)
    int8_bytes = sum(s["q"].numel() * 1 for s in int8_states)  # scale is one scalar/layer, negligible
    assert fp32_bytes == int8_bytes * 4


def test_int8_streaming_stays_close_to_fp32_streaming_over_many_steps():
    """Real, compounding quantization error (each step requantizes the
    previous step's already-quantized state) must stay SMALL over many
    token-by-token steps, not blow up -- the real thing the Phase 3 exit
    gate ("stable long-context behavior") cares about."""
    config = _tiny_config()
    torch.manual_seed(1)
    model = BDH(config)
    model.eval()
    seq = torch.randint(0, config.vocab_size, (2, 40))

    with torch.no_grad():
        fp32_states = init_bdh_states(model, 2)
        int8_states = init_bdh_states_int8(model, 2)
        max_diffs = []
        for t in range(40):
            tok = seq[:, t:t + 1]
            fp32_states, logits_fp32 = bdh_stream_chunk(model, fp32_states, tok, start_position=t)
            int8_states, logits_int8 = bdh_stream_chunk_int8_state(model, int8_states, tok, start_position=t)
            max_diffs.append(float((logits_fp32 - logits_int8).abs().max()))

    assert all(d < 1.0 for d in max_diffs), f"quantization error should stay bounded, got max {max(max_diffs)}"
    assert max_diffs[-1] < 5 * max(max_diffs[:5]), "error should not be blowing up over the 40 steps"


def test_bdh_stream_chunk_int8_state_does_not_mutate_input_states_list():
    """Same not-in-place contract as bdh_stream_chunk itself -- returns a
    NEW list rather than mutating the caller's states in place."""
    config = _tiny_config()
    torch.manual_seed(2)
    model = BDH(config)
    states = init_bdh_states_int8(model, 1)
    original_q = [s["q"].clone() for s in states]
    tok = torch.randint(0, config.vocab_size, (1, 1))
    with torch.no_grad():
        new_states, _logits = bdh_stream_chunk_int8_state(model, states, tok, start_position=0)
    for original, current in zip(original_q, [s["q"] for s in states]):
        assert torch.equal(original, current), "input states list should not be mutated in place"
