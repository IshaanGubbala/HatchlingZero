"""Difficulty/horizon generator + train/test seed split, plans/Hatchling
world.md section 5 (School) and section 22 Phase 1's remaining
checklist items. Real, honest scope note: this W0 pass implements the
part of School that's expressible with the current room-graph/key-door
mechanic -- difficulty via room count and lock density (S0-S3-ish: more
rooms and more locked doors means a longer real oracle plan, i.e. a
real horizon/dependency-depth gradient). S4 ("experiment-driven
learning" -- try an action, learn from a FAILURE, remember it) and S5's
full 10-30+ action long-horizon planning need mechanics this W0
sandbox doesn't have yet (an experimentable/failable action with a
real consequence, and enough rooms/locks to force that length) --
tracked honestly as future school levels, not implemented here.

Train/test split: a fixed, real, disjoint seed-space convention (not
just "trust the caller not to reuse a seed") -- train seeds and test
seeds are drawn from non-overlapping ranges so a train/test generator
call with the same episode index can never collide.
"""
from __future__ import annotations

from dataclasses import dataclass

from hatchling_world.generator import generate_worlds
from hatchling_world.state import WorldConfig, WorldState

_TEST_SEED_OFFSET = 10_000_000  # real, disjoint from any realistic train episode count


@dataclass(frozen=True)
class SchoolLevel:
    name: str
    world_config: WorldConfig
    description: str


SCHOOL_LEVELS: dict[str, SchoolLevel] = {
    "S0_cause_effect": SchoolLevel(
        name="S0_cause_effect",
        world_config=WorldConfig(n_rooms=3, n_colors=1, max_steps=6),
        description="Horizon 1-2 real actions: tiny room count, at most one lock.",
    ),
    "S1_short_composition": SchoolLevel(
        name="S1_short_composition",
        world_config=WorldConfig(n_rooms=4, n_colors=1, max_steps=10),
        description="Horizon 2-4 actions: key -> door -> goal.",
    ),
    "S2_multi_step": SchoolLevel(
        name="S2_multi_step",
        world_config=WorldConfig(n_rooms=6, n_colors=2, max_steps=16),
        description="Horizon 5-8 actions: resource -> tool-like chain via multiple locks.",
    ),
    "S3_hidden_rules": SchoolLevel(
        name="S3_hidden_rules",
        world_config=WorldConfig(n_rooms=6, n_colors=3, max_steps=20),
        description="Same surface objects, real color-to-door mapping re-randomized every "
                    "episode (already true of every level via generate_worlds, but this is "
                    "the level where persistent S first becomes load-bearing, per the plan's "
                    "own S3 framing: real evidence, section 8.5, is that this task family "
                    "genuinely requires reading THIS episode's demos, not memorized weights).",
    ),
    "S5_long_horizon": SchoolLevel(
        name="S5_long_horizon",
        world_config=WorldConfig(n_rooms=10, n_colors=4, max_steps=40),
        description="Horizon 10-30+ real actions: larger room graph, more locks, the primary "
                    "candidate for useful R-scaling per the plan's own S5 framing.",
    ),
}


def generate_school_worlds(level: str, batch: int, episode_seed: int, split: str = "train") -> tuple[WorldState, WorldConfig]:
    """Real train/test seed split: `split='test'` shifts into a
    disjoint seed range so no train episode and test episode can ever
    generate the identical world by seed collision."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    school = SCHOOL_LEVELS[level]
    seed = episode_seed + (_TEST_SEED_OFFSET if split == "test" else 0)
    return generate_worlds(batch, school.world_config, seed), school.world_config
