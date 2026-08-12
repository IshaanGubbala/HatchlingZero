"""Regression tests for reference/hz0h_bdh_vb_torch.py's combined
value-bottleneck + INT8-state functions (Phase 2R-E,
plans/HZ Phase 2R State Redesign Plan.md) -- bdh_vb_stream_chunk_int8_state/
init_bdh_vb_states_int8. Pins down: real 4x additional byte reduction on
top of the value bottleneck's own reduction, and bounded (not exploding)
compounding quantization error over many token-by-token steps.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_vb_torch import (
    BDHVB,
    BDHVBConfig,
    bdh_vb_stream_chunk,
    bdh_vb_stream_chunk_int8_state,
    init_bdh_vb_states,
    init_bdh_vb_states_int8,
)


def _tiny_config(d_state: int = 8) -> BDHVBConfig:
    return BDHVBConfig(n_layer=3, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0, d_state=d_state)


def test_int8_vb_state_bytes_are_4x_smaller_than_fp32_vb_state():
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDHVB(config)
    fp32_states = init_bdh_vb_states(model, batch_size=1)
    int8_states = init_bdh_vb_states_int8(model, batch_size=1)

    fp32_bytes = sum(s.numel() * 4 for s in fp32_states)
    int8_bytes = sum(s["q"].numel() * 1 for s in int8_states)
    assert fp32_bytes == int8_bytes * 4


def test_combined_reduction_matches_d_state_ratio_times_4():
    """Real, compounded reduction: value-bottleneck's own D/d_state ratio,
    times INT8's own 4x -- confirmed via real byte counts, not assumed."""
    config_full = _tiny_config(d_state=32)  # no VB compression
    config_quarter = _tiny_config(d_state=8)  # 4x VB compression
    model_full = BDHVB(config_full)
    model_quarter = BDHVB(config_quarter)

    full_int8_bytes = sum(s["q"].numel() for s in init_bdh_vb_states_int8(model_full, batch_size=1))
    quarter_int8_bytes = sum(s["q"].numel() for s in init_bdh_vb_states_int8(model_quarter, batch_size=1))
    assert full_int8_bytes == quarter_int8_bytes * 4  # the VB ratio, INT8 factor cancels (same for both)


def test_combined_streaming_stays_close_to_fp32_vb_streaming_over_many_steps():
    config = _tiny_config()
    torch.manual_seed(1)
    model = BDHVB(config)
    model.eval()
    seq = torch.randint(0, config.vocab_size, (2, 30))

    with torch.no_grad():
        fp32_states = init_bdh_vb_states(model, 2)
        int8_states = init_bdh_vb_states_int8(model, 2)
        max_diffs = []
        for t in range(30):
            tok = seq[:, t:t + 1]
            fp32_states, logits_fp32 = bdh_vb_stream_chunk(model, fp32_states, tok, start_position=t)
            int8_states, logits_int8 = bdh_vb_stream_chunk_int8_state(model, int8_states, tok, start_position=t)
            max_diffs.append(float((logits_fp32 - logits_int8).abs().max()))

    assert all(d < 1.0 for d in max_diffs), f"error should stay bounded, got max {max(max_diffs)}"
    assert max_diffs[-1] < 10 * max(max_diffs[:5]), "error should not be blowing up over the 30 steps"
