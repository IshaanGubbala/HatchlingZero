"""Procedural, guaranteed-solvable world generation, plans/Hatchling world.md
section 4.3. Real construction (not reject-and-retry): build a random
spanning tree over the rooms, then process tree edges in discovery
order, optionally locking an edge and placing its key in a room that
is PROVABLY already reachable without that edge -- this constructs
solvability directly instead of generating-then-checking. A handful of
extra always-unlocked edges add topology variety without touching the
lock/key logic (unlocked edges can never make a solvable world
unsolvable).

The key-color-to-door mapping is freshly randomized every call (every
episode) -- the real point of this task family (section 4.3): the same
surface objects (rooms, key colors) support a different actual mapping
each time, so a policy has to use the CURRENT episode's persistent
memory S, not memorized weights, to solve it.
"""
from __future__ import annotations

import torch

from hatchling_world.state import WorldConfig, WorldState


def _generate_one(config: WorldConfig, rng: torch.Generator) -> tuple:
    R, C = config.n_rooms, config.n_colors
    door_adj = torch.zeros(R, R, dtype=torch.bool)
    door_locked = torch.zeros(R, R, dtype=torch.bool)
    door_key_color = torch.full((R, R), -1, dtype=torch.long)
    room_keys = torch.zeros(R, C, dtype=torch.long)

    order = (torch.randperm(R, generator=rng)).tolist()
    root = order[0]
    reachable = [root]
    lock_prob = 0.6

    for child in order[1:]:
        # attach `child` to a random already-placed room (spanning tree)
        parent = reachable[torch.randint(0, len(reachable), (1,), generator=rng).item()]
        door_adj[parent, child] = True
        door_adj[child, parent] = True

        if torch.rand(1, generator=rng).item() < lock_prob:
            color = torch.randint(0, C, (1,), generator=rng).item()
            door_locked[parent, child] = True
            door_locked[child, parent] = True
            door_key_color[parent, child] = color
            door_key_color[child, parent] = color
            # place the key in a room already reachable WITHOUT this edge --
            # guarantees the agent can obtain it before needing the door.
            key_room = reachable[torch.randint(0, len(reachable), (1,), generator=rng).item()]
            room_keys[key_room, color] += 1

        reachable.append(child)

    # A few extra always-UNLOCKED shortcut edges for topology variety --
    # cannot break solvability since they only add reachability.
    n_extra = max(0, R // 3)
    for _ in range(n_extra):
        a, b = torch.randint(0, R, (2,), generator=rng).tolist()
        if a != b and not door_adj[a, b]:
            door_adj[a, b] = True
            door_adj[b, a] = True

    agent_room = root
    goal_room = order[-1]  # last room discovered in the spanning tree -- real, furthest guaranteed-solvable target
    return door_adj, door_locked, door_key_color, room_keys, agent_room, goal_room


def generate_worlds(batch: int, config: WorldConfig, seed: int) -> WorldState:
    rng = torch.Generator().manual_seed(seed)
    R, C = config.n_rooms, config.n_colors
    door_adj = torch.zeros(batch, R, R, dtype=torch.bool)
    door_locked = torch.zeros(batch, R, R, dtype=torch.bool)
    door_key_color = torch.full((batch, R, R), -1, dtype=torch.long)
    room_keys = torch.zeros(batch, R, C, dtype=torch.long)
    agent_room = torch.zeros(batch, dtype=torch.long)
    goal_room = torch.zeros(batch, dtype=torch.long)

    for b in range(batch):
        da, dl, dk, rk, a, g = _generate_one(config, rng)
        door_adj[b], door_locked[b], door_key_color[b], room_keys[b] = da, dl, dk, rk
        agent_room[b] = a
        goal_room[b] = g

    return WorldState(
        door_adj=door_adj, door_locked=door_locked, door_key_color=door_key_color,
        room_keys=room_keys, inventory=torch.zeros(batch, C, dtype=torch.long),
        agent_room=agent_room, goal_room=goal_room,
        steps_taken=torch.zeros(batch, dtype=torch.long), done=torch.zeros(batch, dtype=torch.bool),
    )
