"""Procedural example generators for Language Nursery stages L0/L1,
plans/Hatchling world.md section 5. Real, deterministic (seeded),
closed-vocabulary generation -- no real natural-language corpus
involved yet, by design (Stage 0-1 of the curriculum, section 13)."""
from __future__ import annotations

import random

from hatchling_world.language.tokenizer import COLORS, NOUNS, NUMBERS, POSITIONS, SIZES, VERBS_ACTION


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


# Stage L3: fixed size x color combination split. Deliberately a FIXED,
# non-random split (not reseeded per call) so "train" and "test" mean
# the same thing across every run -- the whole point is that these two
# combos are NEVER the supervised target during training, so a good
# held-out score is genuine compositional generalization to an unseen
# (size, color) PAIR, not just a new episode of a pair already seen.
ALL_SIZE_COLOR_COMBOS = [(s, c) for s in SIZES for c in COLORS]
HELD_OUT_COMBOS = [("small", "yellow"), ("large", "green")]
TRAIN_COMBOS = [combo for combo in ALL_SIZE_COLOR_COMBOS if combo not in HELD_OUT_COMBOS]


def _build_compositional_episode(rng: random.Random, n_objects: int, split: str) -> tuple[list[dict], int, str, str]:
    """Shared object-construction for any task where the target is
    identified by a (size, color) PAIR and neither property alone may
    identify it -- both individually collide with a decoy on purpose.
    Returns (objects, target_idx, target_size, target_color); callers
    just choose the instruction wording (L3's bare juxtaposition vs
    L4's explicit "that is X and Y")."""
    if n_objects < 3:
        raise ValueError("needs at least 3 objects (target + one same-color decoy + one same-size decoy)")
    combo_pool = TRAIN_COMBOS if split == "train" else HELD_OUT_COMBOS
    target_size, target_color = rng.choice(combo_pool)

    decoys = [
        {"size": rng.choice([s for s in SIZES if s != target_size]), "color": target_color},
        {"size": target_size, "color": rng.choice([c for c in COLORS if c != target_color])},
    ]
    while len(decoys) < n_objects - 1:
        size, color = rng.choice(TRAIN_COMBOS)
        if (size, color) != (target_size, target_color):
            decoys.append({"size": size, "color": color})

    objects = [{"type": rng.choice(NOUNS), "size": d["size"], "color": d["color"],
                "position": rng.choice(POSITIONS)} for d in decoys]
    objects.append({"type": rng.choice(NOUNS), "size": target_size, "color": target_color,
                     "position": rng.choice(POSITIONS)})
    rng.shuffle(objects)
    target_idx = next(i for i, o in enumerate(objects)
                       if o["size"] == target_size and o["color"] == target_color)
    return objects, target_idx, target_size, target_color


def generate_l3_relation_episode(rng: random.Random, n_objects: int = 4, split: str = "train") -> dict:
    """Stage L3: relations/composition. Neither color nor size alone
    identifies the target -- both individually collide with a decoy on
    purpose -- so "touch the {size} {color} object" is only solvable by
    COMBINING two words, real compositional reference instead of L1's
    single distinguishing property. `split="test"` forces the target's
    (size, color) pair to be one never used as a target during
    `split="train"` generation -- a real held-out COMBINATION test
    (each property individually is seen constantly; the pairing is
    not), not just a held-out episode of a familiar pairing."""
    objects, target_idx, target_size, target_color = _build_compositional_episode(rng, n_objects, split)
    instruction = f"touch the {target_size} {target_color} object"
    return {"objects": objects, "instruction": instruction, "target_idx": target_idx}


def generate_l4_logic_and_episode(rng: random.Random, n_objects: int = 4, split: str = "train") -> dict:
    """Stage L4, logic half: the same compositional-reference task as
    L3, but phrased with an explicit logic word ("that is X and Y")
    instead of bare juxtaposition -- tests whether "and" itself is
    grounded as conjunction (both properties must hold) rather than
    the model merely pattern-matching L3's specific template."""
    objects, target_idx, target_size, target_color = _build_compositional_episode(rng, n_objects, split)
    instruction = f"touch the object that is {target_color} and {target_size}"
    return {"objects": objects, "instruction": instruction, "target_idx": target_idx}


def generate_l4_counting_episode(rng: random.Random, n_objects: int = 4) -> dict:
    """Stage L4, numbers half: grounds numeral WORDS to real quantities
    via a verification task -- "are there {number} {value} objects" --
    rather than treating counting as a hidden classification target
    with no linguistic form. Balanced: the stated number is the TRUE
    count half the time (label=True) and a deliberately wrong count the
    other half (label=False), so the model can't win by always
    answering one way."""
    property_kind = rng.choice(["color", "size"])
    value = rng.choice(COLORS if property_kind == "color" else SIZES)
    objects = [{
        "type": rng.choice(NOUNS),
        "color": rng.choice(COLORS),
        "size": rng.choice(SIZES),
        "position": rng.choice(POSITIONS),
    } for _ in range(n_objects)]
    true_count = sum(1 for o in objects if o[property_kind] == value)

    if rng.random() < 0.5:
        stated_count, label = true_count, True
    else:
        other_counts = [c for c in range(n_objects + 1) if c != true_count]
        stated_count, label = rng.choice(other_counts), False

    instruction = f"are there {NUMBERS[stated_count]} {value} objects"
    return {"objects": objects, "instruction": instruction, "label": label,
            "true_count": true_count, "stated_count": stated_count,
            "property_kind": property_kind, "value": value}
