"""Fixed-shape world state, plans/Hatchling world.md section 4.1:
represent every world with fixed-shape tensors so B parallel worlds
stack into real batched tensors (W_t in R^{B x ...}), no per-world
Python objects in the hot path.

Real, minimal W0 design (section 4.3's own worked example: "colored
keys open different door classes"): a graph of N_ROOMS rooms, some
pairs connected by a door, each door either open or locked requiring
one of N_COLORS key colors. Colored keys are scattered in rooms. The
agent must navigate, pick up keys, and unlock doors to reach a fixed
goal room. The color-to-lock mapping (which key color opens which
specific doors) is re-randomized every episode -- section 4.3's "same
surface objects should support different mappings across episodes so
model weights alone cannot simply memorize the solution."
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WorldConfig:
    n_rooms: int = 6
    n_colors: int = 3
    max_steps: int = 30
    step_cost: float = -0.01
    goal_reward: float = 1.0
    invalid_action_penalty: float = -0.02

    @property
    def n_actions(self) -> int:
        # MOVE(room) x n_rooms + USE_KEY(color, target_room) x n_colors x n_rooms + PICKUP + INSPECT
        # USE_KEY takes an explicit target room -- real bug fixed 2026-09-04:
        # an underspecified USE_KEY(color) (no target) is ambiguous whenever
        # two locked doors adjacent to the same room need the same color;
        # the oracle's search model and the real transition function
        # silently disagreed on which one got unlocked. Explicit targeting
        # removes the ambiguity entirely, same design language as MOVE
        # (most (color, target) combos are simply invalid, agent learns
        # which ones are real from S, exactly like most MOVE targets are).
        return self.n_rooms + self.n_colors * self.n_rooms + 2

    def action_move(self, room: int) -> int:
        return room

    def action_use_key(self, color: int, target_room: int) -> int:
        return self.n_rooms + color * self.n_rooms + target_room

    def decode_use_key(self, action: int) -> tuple[int, int]:
        """Inverse of action_use_key -- (color, target_room)."""
        offset = action - self.n_rooms
        return offset // self.n_rooms, offset % self.n_rooms

    @property
    def action_pickup(self) -> int:
        return self.n_rooms + self.n_colors * self.n_rooms

    @property
    def action_inspect(self) -> int:
        return self.n_rooms + self.n_colors * self.n_rooms + 1


@dataclass
class WorldState:
    """All tensors are batched: leading dim B. Real fixed shapes:
    - door_adj:        (B, R, R) bool  -- symmetric, 1 if a door exists
    - door_locked:      (B, R, R) bool  -- 1 if that door is currently locked
    - door_key_color:   (B, R, R) long  -- which color unlocks it, -1 if no door
    - room_keys:        (B, R, C) long  -- count of each key color sitting in each room
    - inventory:        (B, C)    long  -- count of each key color the agent holds
    - agent_room:       (B,)      long
    - goal_room:        (B,)      long
    - steps_taken:      (B,)      long
    - done:             (B,)      bool
    """
    door_adj: torch.Tensor
    door_locked: torch.Tensor
    door_key_color: torch.Tensor
    room_keys: torch.Tensor
    inventory: torch.Tensor
    agent_room: torch.Tensor
    goal_room: torch.Tensor
    steps_taken: torch.Tensor
    done: torch.Tensor

    def batch_size(self) -> int:
        return self.agent_room.shape[0]

    def to(self, device) -> "WorldState":
        return WorldState(**{k: v.to(device) for k, v in self.__dict__.items()})

    def clone(self) -> "WorldState":
        return WorldState(**{k: v.clone() for k, v in self.__dict__.items()})

    def index(self, i: int) -> "WorldState":
        """A real, single-world (batch=1) view -- used by the oracle
        (BFS is per-instance, not batched) and the live visualizer."""
        return WorldState(**{k: v[i:i + 1] for k, v in self.__dict__.items()})
