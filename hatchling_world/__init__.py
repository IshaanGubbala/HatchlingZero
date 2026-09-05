"""Hatchling World -- HZ-World-0 minimal procedural sandbox, plans/Hatchling world.md
section 4. A deterministic, procedurally-generated, fixed-shape,
vectorizable room-graph environment: colored keys unlock colored
doors, the agent must navigate + collect keys + unlock doors to reach
a goal room, and the key-color-to-door-class mapping is re-randomized
every episode so weights alone cannot memorize a fixed solution."""
from hatchling_world.state import WorldState, WorldConfig
from hatchling_world.generator import generate_worlds
from hatchling_world.transition import step
from hatchling_world.oracle import solve
from hatchling_world.rewards import compute_reward

__all__ = ["WorldState", "WorldConfig", "generate_worlds", "step", "solve", "compute_reward"]
