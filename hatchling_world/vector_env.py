"""Thin, real, batched Gym-like wrapper -- plans/Hatchling world.md
section 12 SPEED-W0/W3: many worlds as one batched tensor, transitions
applied in parallel, never one Python environment per agent."""
from __future__ import annotations

import torch

from hatchling_world.state import WorldConfig, WorldState
from hatchling_world.generator import generate_worlds
from hatchling_world.transition import step as env_step


class HatchlingWorldVectorEnv:
    def __init__(self, batch: int, config: WorldConfig | None = None, seed: int = 0):
        self.batch = batch
        self.config = config or WorldConfig()
        self.seed = seed
        self.state: WorldState = generate_worlds(batch, self.config, seed)

    def reset(self, seed: int | None = None) -> WorldState:
        self.seed = seed if seed is not None else self.seed + 1
        self.state = generate_worlds(self.batch, self.config, self.seed)
        return self.state

    def step(self, action: torch.Tensor) -> tuple[WorldState, torch.Tensor, torch.Tensor]:
        self.state, reward, done = env_step(self.state, action, self.config)
        return self.state, reward, done
