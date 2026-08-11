"""Regression tests for scripts/hz0h_state_memory_analysis.py -- pins
down the real, measured finding in
docs/restart/hz0h_phase2_streaming_state_size_results.md: BDH's fixed
synaptic state is already larger than the model's own weights at every
scale tested, and the crossover context length (where a real Transformer
KV-cache would cost the same memory) is well beyond any context length
this project has benchmarked so far.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from hz0h_state_memory_analysis import crossover_context_length, state_bytes_per_batch_item
from reference.hz0h_bdh_torch import BDHConfig


def test_state_bytes_scale_with_layers_and_m_and_d_squared():
    """State elements per layer = m * D^2 (see the module docstring's
    derivation) -- doubling m or n_layer should exactly double total
    state bytes; doubling D should roughly quadruple it."""
    base = BDHConfig(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)
    double_layers = BDHConfig(n_layer=4, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)
    double_m = BDHConfig(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=16, vocab_size=32, dropout=0.0)
    double_d = BDHConfig(n_layer=2, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)

    base_bytes = state_bytes_per_batch_item(base)
    assert state_bytes_per_batch_item(double_layers) == base_bytes * 2
    assert state_bytes_per_batch_item(double_m) == base_bytes * 2
    assert state_bytes_per_batch_item(double_d) == base_bytes * 4


def test_state_exceeds_model_weights_at_the_pilot_scales():
    """Real, measured finding this session: at every matched pilot scale
    (~5M/25M/71M), BDH's fixed state (batch=1) is already bigger than
    the model's own weights -- not the "small fixed footprint" a naive
    reading of "O(1) state" might suggest."""
    import torch
    from reference.hz0h_bdh_torch import BDH

    configs = [
        BDHConfig(n_layer=6, n_embd=256, n_head=4, mlp_internal_dim_multiplier=24, vocab_size=256, dropout=0.0),
        BDHConfig(n_layer=8, n_embd=512, n_head=8, mlp_internal_dim_multiplier=32, vocab_size=256, dropout=0.0),
        BDHConfig(n_layer=10, n_embd=768, n_head=12, mlp_internal_dim_multiplier=40, vocab_size=256, dropout=0.0),
    ]
    for config in configs:
        model = BDH(config)
        param_bytes = sum(p.numel() for p in model.parameters()) * 4
        state_bytes = state_bytes_per_batch_item(config)
        assert state_bytes > param_bytes, f"expected state ({state_bytes}) to exceed weights ({param_bytes}) at D={config.n_embd}"


def test_crossover_context_length_grows_with_model_width():
    """Real finding: the context length at which BDH's state becomes
    memory-cheaper than a real KV-cache grows with D (crossover = m*D/2),
    the same unfavorable-at-scale direction as the decode-speed finding
    in docs/restart/hz0h_phase1_kv_cache_bdh_results.md."""
    small = BDHConfig(n_layer=4, n_embd=128, n_head=4, mlp_internal_dim_multiplier=16, vocab_size=32, dropout=0.0)
    large = BDHConfig(n_layer=4, n_embd=256, n_head=4, mlp_internal_dim_multiplier=16, vocab_size=32, dropout=0.0)
    assert crossover_context_length(large) > crossover_context_length(small)


def test_crossover_context_length_matches_closed_form():
    """crossover = m * D / 2, derived in the module docstring -- confirm
    the empirically-measured (via real tensor byte counts) crossover
    matches the closed-form prediction, not just "some number.\""""
    config = BDHConfig(n_layer=6, n_embd=256, n_head=4, mlp_internal_dim_multiplier=24, vocab_size=32, dropout=0.0)
    measured = crossover_context_length(config)
    closed_form = config.mlp_internal_dim_multiplier * config.n_embd / 2
    assert measured == pytest.approx(closed_form, rel=1e-6)
