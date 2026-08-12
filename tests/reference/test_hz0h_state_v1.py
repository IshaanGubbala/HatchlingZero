"""Regression test for reference/hz0h_state_v1.py -- pins down the
locked HZ-State-v1 configuration (plans/HZ Integrated Candidate Plan.md
Step 1): d_state = n_embd // 4, matching the safe (not the extreme)
point on the reassignment-task compression cliff.
"""
from __future__ import annotations

import pytest

from reference.hz0h_state_v1 import hz_state_v1_config


def test_d_state_is_one_quarter_of_n_embd():
    config = hz_state_v1_config(n_layer=6, n_embd=256, n_head=4, mlp_internal_dim_multiplier=24, vocab_size=256)
    assert config.d_state == 64


def test_rejects_n_embd_not_divisible_by_four():
    with pytest.raises(ValueError):
        hz_state_v1_config(n_layer=2, n_embd=30, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32)


def test_builds_a_real_bdhvb_model():
    from reference.hz0h_bdh_vb_torch import BDHVB

    config = hz_state_v1_config(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32)
    model = BDHVB(config)
    assert model.P.shape == (32, 8)
    assert model.O.shape == (8, 32)
