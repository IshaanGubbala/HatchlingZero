"""Reward computation, plans/Hatchling world.md section 4/W0: a real,
verifiable, sparse-plus-small-shaping reward -- goal reward on success,
small step cost otherwise, an invalid-action penalty distinguishable
from a normal step. No reward for "looking thoughtful" (section
7/W5's own explicit rule) -- nothing here rewards anything but reaching
the real goal room, efficiently, without invalid actions."""
from __future__ import annotations

import torch

from hatchling_world.state import WorldConfig


def compute_reward(reached_goal: torch.Tensor, invalid: torch.Tensor, was_done: torch.Tensor,
                    config: WorldConfig) -> torch.Tensor:
    reward = torch.zeros_like(reached_goal, dtype=torch.float32)
    reward = torch.where(reached_goal & ~was_done, torch.full_like(reward, config.goal_reward), reward)
    reward = torch.where(invalid & ~was_done, torch.full_like(reward, config.invalid_action_penalty), reward)
    reward = torch.where((~invalid) & (~reached_goal) & (~was_done), torch.full_like(reward, config.step_cost), reward)
    reward = torch.where(was_done, torch.zeros_like(reward), reward)
    return reward
