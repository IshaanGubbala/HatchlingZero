"""Real curriculum/train-test-split tests, plans/Hatchling world.md
section 5 (School) and section 22 Phase 1's remaining checklist items."""
from __future__ import annotations

from hatchling_world.curriculum import SCHOOL_LEVELS, generate_school_worlds
from hatchling_world.oracle import is_solvable


def test_every_school_level_is_solvable():
    for name in SCHOOL_LEVELS:
        for seed in range(10):
            state, config = generate_school_worlds(name, batch=2, episode_seed=seed)
            for i in range(2):
                assert is_solvable(state, config, index=i), f"{name} seed={seed} world={i} unsolvable"


def test_difficulty_increases_room_count_across_levels():
    """Real, monotonic sanity check that the school levels actually
    form a difficulty ladder, not just five arbitrary configs."""
    order = ["S0_cause_effect", "S1_short_composition", "S2_multi_step", "S3_hidden_rules", "S5_long_horizon"]
    room_counts = [SCHOOL_LEVELS[name].world_config.n_rooms for name in order]
    assert room_counts == sorted(room_counts)


def test_train_and_test_splits_never_collide():
    train_state, config = generate_school_worlds("S2_multi_step", batch=1, episode_seed=5, split="train")
    test_state, _ = generate_school_worlds("S2_multi_step", batch=1, episode_seed=5, split="test")
    import torch
    assert not torch.equal(train_state.door_adj, test_state.door_adj) or \
           not torch.equal(train_state.agent_room, test_state.agent_room) or \
           not torch.equal(train_state.goal_room, test_state.goal_room)


def test_invalid_split_name_raises():
    import pytest
    with pytest.raises(ValueError):
        generate_school_worlds("S0_cause_effect", batch=1, episode_seed=0, split="bogus")
