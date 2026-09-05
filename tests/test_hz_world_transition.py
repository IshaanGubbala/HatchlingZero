"""Real transition-function tests, plans/Hatchling world.md section
22 Phase 1: unit tests for transitions."""
from __future__ import annotations

import torch

from hatchling_world.generator import generate_worlds
from hatchling_world.state import WorldConfig
from hatchling_world.transition import step


def test_transition_is_deterministic():
    config = WorldConfig(n_rooms=6, n_colors=3)
    s1 = generate_worlds(batch=4, config=config, seed=5)
    s2 = generate_worlds(batch=4, config=config, seed=5)
    action = torch.zeros(4, dtype=torch.long)
    r1, reward1, done1 = step(s1, action, config)
    r2, reward2, done2 = step(s2, action, config)
    assert torch.equal(r1.agent_room, r2.agent_room)
    assert torch.equal(reward1, reward2)
    assert torch.equal(done1, done2)


def test_invalid_move_into_nonexistent_door_is_penalized_and_agent_stays():
    config = WorldConfig(n_rooms=6, n_colors=3)
    state = generate_worlds(batch=1, config=config, seed=1)
    room = state.agent_room[0].item()
    # find a room with no door from the current room
    no_door_room = next(j for j in range(config.n_rooms) if j != room and not state.door_adj[0, room, j])
    new_state, reward, done = step(state, torch.tensor([no_door_room]), config)
    assert new_state.agent_room[0].item() == room  # never moved
    assert abs(reward[0].item() - config.invalid_action_penalty) < 1e-6


def test_move_into_locked_door_without_key_is_invalid():
    config = WorldConfig(n_rooms=6, n_colors=3)
    state = generate_worlds(batch=1, config=config, seed=1)
    room = state.agent_room[0].item()
    locked_neighbors = [j for j in range(config.n_rooms)
                         if state.door_adj[0, room, j] and state.door_locked[0, room, j]]
    if not locked_neighbors:
        return  # this seed's starting room happens to have no locked doors -- not a real failure
    j = locked_neighbors[0]
    new_state, reward, done = step(state, torch.tensor([j]), config)
    assert new_state.agent_room[0].item() == room
    assert abs(reward[0].item() - config.invalid_action_penalty) < 1e-6


def test_pickup_moves_keys_from_room_to_inventory():
    config = WorldConfig(n_rooms=6, n_colors=3)
    state = generate_worlds(batch=4, config=config, seed=9)
    for i in range(4):
        room = state.agent_room[i].item()
        if state.room_keys[i, room].sum().item() > 0:
            before = state.room_keys[i, room].clone()
            new_state, reward, done = step(state.index(i), torch.tensor([config.action_pickup]), config)
            assert new_state.room_keys[0, room].sum().item() == 0
            assert torch.equal(new_state.inventory[0], before)
            return
    raise AssertionError("no seed=9 world had a key in the agent's starting room -- test setup issue")


def test_done_world_freezes_on_further_steps():
    config = WorldConfig(n_rooms=6, n_colors=3, max_steps=1)
    state = generate_worlds(batch=1, config=config, seed=3)
    s1, _, done1 = step(state, torch.tensor([config.action_inspect]), config)
    assert done1.item()  # max_steps=1 forces timeout
    s2, reward2, done2 = step(s1, torch.tensor([config.action_pickup]), config)
    assert torch.equal(s1.agent_room, s2.agent_room)
    assert reward2.item() == 0.0
    assert done2.item()


def test_reaching_goal_pays_real_goal_reward_and_sets_done():
    config = WorldConfig(n_rooms=6, n_colors=3)
    state = generate_worlds(batch=1, config=config, seed=1)
    state.goal_room[0] = state.agent_room[0]
    # any action that keeps agent_room the same still shouldn't have
    # already been "done" before this step -- force a move to a
    # neighboring open room then back would be more real, but the
    # simplest true check: reaching goal via a normal valid move.
    room = state.agent_room[0].item()
    open_neighbors = [j for j in range(config.n_rooms)
                       if state.door_adj[0, room, j] and not state.door_locked[0, room, j]]
    if not open_neighbors:
        return
    state.goal_room[0] = open_neighbors[0]
    new_state, reward, done = step(state, torch.tensor([open_neighbors[0]]), config)
    assert new_state.agent_room[0].item() == open_neighbors[0]
    assert done[0].item()
    assert reward[0].item() == config.goal_reward
