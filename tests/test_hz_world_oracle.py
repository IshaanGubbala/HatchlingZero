"""Real oracle-correctness tests, plans/Hatchling world.md section
1.2/22 Phase 1: "is the task actually solvable? / unit tests for
transitions and solvability."""
from __future__ import annotations

import torch

from hatchling_world.generator import generate_worlds
from hatchling_world.oracle import solve, is_solvable
from hatchling_world.state import WorldConfig
from hatchling_world.transition import step


def test_generated_worlds_are_always_solvable():
    config = WorldConfig(n_rooms=6, n_colors=3, max_steps=30)
    for seed in range(20):
        state = generate_worlds(batch=4, config=config, seed=seed)
        for i in range(4):
            assert is_solvable(state, config, index=i), f"seed={seed} world={i} unsolvable"


def test_oracle_plan_actually_reaches_the_goal_in_the_real_env():
    """The real, important check: replay the oracle's OWN plan through
    the actual transition function and confirm it reaches the goal --
    not just that BFS found *a* path in its own internal model, but
    that the two agree exactly."""
    config = WorldConfig(n_rooms=6, n_colors=3, max_steps=30)
    state_batch = generate_worlds(batch=8, config=config, seed=123)
    for i in range(8):
        plan = solve(state_batch, config, index=i)
        assert plan is not None
        s = state_batch.index(i)
        total_reward = 0.0
        for a in plan:
            s, r, done = step(s, torch.tensor([a]), config)
            total_reward += r.item()
        assert s.agent_room.item() == s.goal_room.item(), "oracle plan did not reach the goal in the real env"
        assert s.done.item()
        assert total_reward > 0  # real goal reward must have been paid out


def test_agent_already_at_goal_returns_empty_plan():
    config = WorldConfig(n_rooms=6, n_colors=3)
    state = generate_worlds(batch=1, config=config, seed=7)
    state.goal_room[0] = state.agent_room[0]
    assert solve(state, config, index=0) == []
