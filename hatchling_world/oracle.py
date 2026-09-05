"""Real BFS oracle solver, plans/Hatchling world.md section 1.2/4.1's
"is the task actually solvable? / is the oracle planner correct?"
rescue-ladder check, and section 7/W2's source of expert trajectories
for behavior cloning. Searches the EXACT same state transitions as
`hatchling_world.transition.step` (room/inventory/door-lock state),
so a returned plan is guaranteed to be a real, valid action sequence
in the actual environment, not an approximation."""
from __future__ import annotations

from collections import deque

from hatchling_world.state import WorldConfig, WorldState


def solve(state: WorldState, config: WorldConfig, index: int = 0) -> list[int] | None:
    """Shortest real action sequence from world `index` (in a possibly
    batched WorldState) to its goal room, or None if unsolvable."""
    R, C = config.n_rooms, config.n_colors
    door_adj = state.door_adj[index].tolist()
    door_key_color = state.door_key_color[index].tolist()
    room_keys0 = tuple(tuple(row) for row in state.room_keys[index].tolist())
    door_locked0 = tuple(tuple(row) for row in state.door_locked[index].tolist())
    agent_room0 = int(state.agent_room[index].item())
    goal_room = int(state.goal_room[index].item())
    inventory0 = tuple(0 for _ in range(C))

    start = (agent_room0, room_keys0, inventory0, door_locked0)
    if agent_room0 == goal_room:
        return []

    visited = {start}
    queue = deque([(start, [])])

    while queue:
        (room, room_keys, inventory, door_locked), path = queue.popleft()
        if len(path) >= config.max_steps:
            continue

        for j in range(R):
            if door_adj[room][j] and j != room and not door_locked[room][j]:
                if j == goal_room:
                    return path + [config.action_move(j)]
                nxt = (j, room_keys, inventory, door_locked)
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [config.action_move(j)]))

        if any(k > 0 for k in room_keys[room]):
            new_room_keys = [list(r) for r in room_keys]
            new_inv = list(inventory)
            for c in range(C):
                new_inv[c] += new_room_keys[room][c]
                new_room_keys[room][c] = 0
            nxt = (room, tuple(tuple(r) for r in new_room_keys), tuple(new_inv), door_locked)
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [config.action_pickup]))

        for c in range(C):
            if inventory[c] > 0:
                for j in range(R):
                    if door_locked[room][j] and door_key_color[room][j] == c:
                        new_door_locked = [list(r) for r in door_locked]
                        new_door_locked[room][j] = False
                        new_door_locked[j][room] = False
                        new_inv = list(inventory)
                        new_inv[c] -= 1
                        nxt = (room, room_keys, tuple(new_inv), tuple(tuple(r) for r in new_door_locked))
                        if nxt not in visited:
                            visited.add(nxt)
                            queue.append((nxt, path + [config.action_use_key(c, j)]))

    return None


def is_solvable(state: WorldState, config: WorldConfig, index: int = 0) -> bool:
    return solve(state, config, index) is not None
