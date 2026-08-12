"""Regression tests for reference/hz0h_bdh_gs_torch.py (HZ-BDH-GS,
Grouped Synaptic State, Phase 2R-C, plans/HZ Phase 2R State Redesign Plan.md).
Pins down: n_groups == n_layer (no sharing) is mathematically IDENTICAL
to reference/hz0h_bdh_torch.py's own bdh_stream_chunk (the degenerate-
case sanity check that validates the whole implementation), state bytes
really do shrink with fewer groups, and layer_group_assignment matches
the user's own example.
"""
from __future__ import annotations

import pytest
import torch

from reference.hz0h_bdh_gs_torch import (
    bdh_grouped_stream_chunk,
    init_bdh_grouped_states,
    layer_group_assignment,
)
from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_chunk, init_bdh_states


def _tiny_config() -> BDHConfig:
    return BDHConfig(n_layer=6, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)


def test_layer_group_assignment_matches_user_example():
    assert layer_group_assignment(6, 2) == [0, 0, 0, 1, 1, 1]
    assert layer_group_assignment(6, 3) == [0, 0, 1, 1, 2, 2]
    assert layer_group_assignment(6, 1) == [0, 0, 0, 0, 0, 0]
    assert layer_group_assignment(6, 6) == [0, 1, 2, 3, 4, 5]


def test_layer_group_assignment_rejects_invalid_n_groups():
    with pytest.raises(ValueError):
        layer_group_assignment(6, 0)
    with pytest.raises(ValueError):
        layer_group_assignment(6, 7)


def test_degenerate_n_groups_equals_n_layer_matches_bdh_stream_chunk_exactly():
    """The whole implementation's real validation: with one group per
    layer (no sharing at all), grouped streaming must be mathematically
    IDENTICAL to the real, already-tested bdh_stream_chunk -- not just
    similar."""
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDH(config)
    model.eval()
    seq = torch.randint(0, config.vocab_size, (2, 15))

    with torch.no_grad():
        states = init_bdh_states(model, 2)
        group_states = init_bdh_grouped_states(model, n_groups=config.n_layer, batch_size=2)
        for t in range(15):
            tok = seq[:, t:t + 1]
            states, logits_a = bdh_stream_chunk(model, states, tok, start_position=t)
            group_states, logits_b = bdh_grouped_stream_chunk(model, group_states, tok, start_position=t, n_groups=config.n_layer)
            assert torch.allclose(logits_a, logits_b, atol=1e-5), f"diverged at step {t}"


def test_a_single_full_sequence_forward_is_unaffected_by_grouping():
    """Real structural property the whole design relies on: a state
    starts at zero, so a single-chunk full-sequence call (no prior
    state) gives the SAME result regardless of n_groups -- grouping only
    matters across multiple streaming calls."""
    config = _tiny_config()
    torch.manual_seed(1)
    model = BDH(config)
    model.eval()
    seq = torch.randint(0, config.vocab_size, (2, 10))

    with torch.no_grad():
        logits_full, _ = model(seq)
        for n_groups in (1, 2, 3, 6):
            group_states = init_bdh_grouped_states(model, n_groups=n_groups, batch_size=2)
            _states, logits_grouped = bdh_grouped_stream_chunk(model, group_states, seq, start_position=0, n_groups=n_groups)
            assert torch.allclose(logits_full, logits_grouped, atol=1e-4), f"n_groups={n_groups} diverged from full forward"


def test_fewer_groups_uses_less_state_memory():
    config = _tiny_config()
    torch.manual_seed(2)
    model = BDH(config)
    states_full = init_bdh_grouped_states(model, n_groups=6, batch_size=1)
    states_grouped = init_bdh_grouped_states(model, n_groups=2, batch_size=1)
    assert sum(s.numel() for s in states_full) == sum(s.numel() for s in states_grouped) * 3


def test_grouping_actually_changes_output_when_state_is_nonzero():
    """Real regression guard: after streaming a real prefix (nonzero
    state), grouped and ungrouped decoding SHOULD diverge for n_groups <
    n_layer -- if they didn't, grouping wouldn't be doing anything and
    this whole file would be a no-op."""
    config = _tiny_config()
    torch.manual_seed(3)
    model = BDH(config)
    model.eval()
    prefix = torch.randint(0, config.vocab_size, (1, 8))
    query = torch.randint(0, config.vocab_size, (1, 1))

    with torch.no_grad():
        states = init_bdh_states(model, 1)
        states, _ = bdh_stream_chunk(model, states, prefix, start_position=0)
        _states, logits_ungrouped = bdh_stream_chunk(model, states, query, start_position=8)

        group_states = init_bdh_grouped_states(model, n_groups=2, batch_size=1)
        group_states, _ = bdh_grouped_stream_chunk(model, group_states, prefix, start_position=0, n_groups=2)
        _gs, logits_grouped = bdh_grouped_stream_chunk(model, group_states, query, start_position=8, n_groups=2)

    assert not torch.allclose(logits_ungrouped, logits_grouped, atol=1e-3)
