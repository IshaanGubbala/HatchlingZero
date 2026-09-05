"""Real batched transition function, plans/Hatchling world.md section
4.2's fixed small action vocabulary, applied to ALL B worlds in one
vectorized call -- no per-world Python loop in the hot path (section
11/12's "vectorize worlds, don't run one Python env per agent").

Action id layout (see WorldConfig): MOVE(room) for room in
[0, n_rooms), USE_KEY(color, target_room) for color in [0, n_colors)
x target_room in [0, n_rooms), then PICKUP, then INSPECT. USE_KEY
takes an explicit target room -- real bug fixed 2026-09-04: an
underspecified USE_KEY(color) is ambiguous whenever two locked doors
adjacent to the same room need the same color.
"""
from __future__ import annotations

import torch

from hatchling_world.state import WorldConfig, WorldState
from hatchling_world.rewards import compute_reward


def step(state: WorldState, action: torch.Tensor, config: WorldConfig) -> tuple[WorldState, torch.Tensor, torch.Tensor]:
    """action: (B,) long. Returns (new_state, reward: (B,) float, done: (B,) bool)."""
    B, R, C = state.batch_size(), config.n_rooms, config.n_colors
    device = state.agent_room.device
    was_done = state.done.clone()

    new_agent_room = state.agent_room.clone()
    new_inventory = state.inventory.clone()
    new_room_keys = state.room_keys.clone()
    new_door_locked = state.door_locked.clone()
    invalid = torch.zeros(B, dtype=torch.bool, device=device)

    use_key_end = R + C * R
    is_move = action < R
    is_use_key = (action >= R) & (action < use_key_end)
    is_pickup = action == config.action_pickup
    # action_inspect: real no-op, no branch needed.

    # -- MOVE --
    if is_move.any():
        idx = is_move.nonzero(as_tuple=True)[0]
        target_room = action[idx]
        cur_room = state.agent_room[idx]
        has_door = state.door_adj[idx, cur_room, target_room]
        locked = state.door_locked[idx, cur_room, target_room]
        same_room = target_room == cur_room
        valid = has_door & (~locked) & (~same_room)
        new_agent_room[idx] = torch.where(valid, target_room, cur_room)
        invalid[idx] = ~valid

    # -- USE_KEY(color, target_room) -- explicit target, no ambiguity --
    if is_use_key.any():
        idx = is_use_key.nonzero(as_tuple=True)[0]
        offset = action[idx] - R
        color = offset // R
        target_room = offset % R
        cur_room = state.agent_room[idx]
        has_key = new_inventory[idx, color] > 0
        is_locked = state.door_locked[idx, cur_room, target_room]
        right_color = state.door_key_color[idx, cur_room, target_room] == color
        valid = has_key & is_locked & right_color
        for k in range(idx.shape[0]):
            if valid[k]:
                b = idx[k].item()
                j = target_room[k].item()
                r = cur_room[k].item()
                new_door_locked[b, r, j] = False
                new_door_locked[b, j, r] = False
                new_inventory[b, color[k].item()] -= 1
        invalid[idx] = ~valid

    # -- PICKUP --
    if is_pickup.any():
        idx = is_pickup.nonzero(as_tuple=True)[0]
        cur_room = state.agent_room[idx]
        keys_here = new_room_keys[idx, cur_room]           # (n, C)
        got_any = keys_here.sum(dim=-1) > 0
        new_inventory[idx, :] += keys_here
        new_room_keys[idx, cur_room, :] = 0
        invalid[idx] = ~got_any

    reached_goal = new_agent_room == state.goal_room
    new_steps = state.steps_taken + 1
    timed_out = new_steps >= config.max_steps
    new_done = was_done | reached_goal | timed_out

    reward = compute_reward(reached_goal, invalid, was_done, config)

    # worlds already done before this call stay frozen (standard vec-env
    # convention -- caller resets/replaces them, this function never does).
    freeze = was_done
    new_agent_room = torch.where(freeze, state.agent_room, new_agent_room)
    new_inventory = torch.where(freeze.unsqueeze(-1), state.inventory, new_inventory)
    new_room_keys = torch.where(freeze.view(-1, 1, 1), state.room_keys, new_room_keys)
    new_door_locked = torch.where(freeze.view(-1, 1, 1), state.door_locked, new_door_locked)
    new_steps = torch.where(freeze, state.steps_taken, new_steps)
    new_done = torch.where(freeze, was_done, new_done)

    new_state = WorldState(
        door_adj=state.door_adj, door_locked=new_door_locked, door_key_color=state.door_key_color,
        room_keys=new_room_keys, inventory=new_inventory, agent_room=new_agent_room,
        goal_room=state.goal_room, steps_taken=new_steps, done=new_done,
    )
    return new_state, reward, new_done
