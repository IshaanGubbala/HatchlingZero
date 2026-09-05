"""Real vectorized-env tests, plans/Hatchling world.md section 22
Phase 1: unit tests for the batched transition engine end to end."""
from __future__ import annotations

import torch

from hatchling_world.state import WorldConfig
from hatchling_world.vector_env import HatchlingWorldVectorEnv


def test_reset_produces_correct_batch_shapes():
    config = WorldConfig(n_rooms=6, n_colors=3)
    env = HatchlingWorldVectorEnv(batch=5, config=config, seed=0)
    s = env.state
    assert s.agent_room.shape == (5,)
    assert s.door_adj.shape == (5, 6, 6)
    assert s.room_keys.shape == (5, 6, 3)
    assert s.inventory.shape == (5, 3)


def test_worlds_in_a_batch_are_independent():
    config = WorldConfig(n_rooms=6, n_colors=3)
    env = HatchlingWorldVectorEnv(batch=8, config=config, seed=42)
    # real, not-all-identical check -- with 8 independently generated
    # worlds it would be a real bug (not chance) if they were all equal
    assert not all(torch.equal(env.state.door_adj[0], env.state.door_adj[i]) for i in range(1, 8))


def test_full_episode_runs_to_completion_via_inspect_actions():
    config = WorldConfig(n_rooms=6, n_colors=3, max_steps=5)
    env = HatchlingWorldVectorEnv(batch=3, config=config, seed=1)
    action = torch.full((3,), config.action_inspect, dtype=torch.long)
    done_ever = torch.zeros(3, dtype=torch.bool)
    for _ in range(config.max_steps):
        state, reward, done = env.step(action)
        done_ever |= done
    assert done_ever.all()  # max_steps timeout must terminate every world
