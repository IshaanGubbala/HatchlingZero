"""Procedural example generators for Language Nursery stages L0/L1,
plans/Hatchling world.md section 5. Real, deterministic (seeded),
closed-vocabulary generation -- no real natural-language corpus
involved yet, by design (Stage 0-1 of the curriculum, section 13)."""
from __future__ import annotations

import random

from hatchling_world.language.tokenizer import COLORS, NOUNS, POSITIONS, SIZES, VERBS_ACTION


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


def apply_verb(verb: str, held: bool, opened: bool, position: str) -> tuple[bool, bool, str]:
    """Stage L2's real "verb meaning": each verb is a real, distinct
    state transition function, not a text label. This is the ONE place
    that defines what push/pickup/drop/open/close actually do -- both
    the generator and any future oracle/env reuse of these verbs must
    call through here so meaning stays a single source of truth.

    Returns the post-action (held, opened, position). Verbs a real
    world would reject in the given pre-state (e.g. "open" on an
    already-open object) are still applied deterministically (no-op /
    idempotent) -- the point of L2 is a CORRECT resulting state, not
    modeling failure semantics yet (that's a later-stage concern)."""
    if verb == "push":
        position = "right" if position == "left" else "left"
    elif verb == "pickup":
        held = True
    elif verb == "drop":
        held = False
    elif verb == "open":
        opened = True
    elif verb == "close":
        opened = False
    else:
        raise ValueError(f"unknown verb: {verb}")
    return held, opened, position


def generate_l2_verb_episode(rng: random.Random, n_objects: int = 4) -> dict:
    """Stage L2: verbs through consequences. Objects again get UNIQUE
    colors (same trick as L1) so "push the {color} object" always names
    exactly one real target -- isolates "did it learn what push DOES"
    from "did it find the right object" (already validated by L1)."""
    colors = rng.sample(COLORS, k=min(n_objects, len(COLORS)))
    while len(colors) < n_objects:
        colors.append(rng.choice(COLORS))
    rng.shuffle(colors)

    objects = []
    for color in colors:
        objects.append({
            "type": rng.choice(NOUNS),
            "color": color,
            "size": rng.choice(SIZES),
            "position": rng.choice(POSITIONS),
            "held": rng.random() < 0.5,
            "opened": rng.random() < 0.5,
        })

    target_idx = rng.randrange(len(objects))
    target = objects[target_idx]
    verb = rng.choice(VERBS_ACTION)
    instruction = f"{verb} the {target['color']} object"

    held_after, opened_after, position_after = apply_verb(
        verb, target["held"], target["opened"], target["position"])

    return {
        "objects": objects,
        "instruction": instruction,
        "target_idx": target_idx,
        "verb": verb,
        "held_after": held_after,
        "opened_after": opened_after,
        "position_after": position_after,
    }
