"""Procedural example generators for Language Nursery stages L0/L1,
plans/Hatchling world.md section 5. Real, deterministic (seeded),
closed-vocabulary generation -- no real natural-language corpus
involved yet, by design (Stage 0-1 of the curriculum, section 13)."""
from __future__ import annotations

import random

from hatchling_world.language.tokenizer import COLORS, NOUNS, POSITIONS, SIZES


def generate_l0_sentence(rng: random.Random) -> str:
    """Stage L0: pure self-supervised text, no world state attached.
    Two real template families straight from the plan's own examples:
    "the X is Y" and "the Y X <verb>"."""
    noun = rng.choice(NOUNS)
    color = rng.choice(COLORS)
    if rng.random() < 0.5:
        return f"the {noun} is {color}"
    verb = "moves" if rng.random() < 0.5 else "is still"
    return f"the {color} {noun} {verb}"


def generate_l1_grounding_episode(rng: random.Random, n_objects: int = 4) -> dict:
    """Stage L1: real world state + a behavioral grounding instruction.
    Objects are guaranteed to have UNIQUE colors among the set, so
    "touch the {color} object" always identifies exactly one real
    target -- matches the plan's own worked example verbatim."""
    colors = rng.sample(COLORS, k=min(n_objects, len(COLORS)))
    while len(colors) < n_objects:
        colors.append(rng.choice(COLORS))  # only reached if n_objects > len(COLORS)
    rng.shuffle(colors)

    objects = []
    for color in colors:
        objects.append({
            "type": rng.choice(NOUNS),
            "color": color,
            "size": rng.choice(SIZES),
            "position": rng.choice(POSITIONS),
        })

    target_idx = rng.randrange(len(objects))
    target = objects[target_idx]
    instruction = f"touch the {target['color']} object"

    return {"objects": objects, "instruction": instruction, "target_idx": target_idx}
