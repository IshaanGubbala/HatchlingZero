"""Procedural example generators for School-0, plans/Hatchling world.md
section 8.2. Real, deterministic (seeded), closed-vocabulary generation,
same discipline as hatchling_world.language.nursery_generator."""
from __future__ import annotations

import random

from hatchling_world.language.tokenizer import COLORS, NUMBERS, SIZES, ENTITY_WORDS

# Fixed, non-reseeded held-out split over the 5x5 addition grid
# (operands 0-4, matching NUMBERS' zero-four range so both operands and
# the worked-example phrasing stay in vocabulary) -- same discipline as
# nursery_generator.HELD_OUT_COMBOS: these 5 (a, b) pairs are NEVER used
# as a training example, so a good held-out score is genuine arithmetic
# generalization to an unseen operand pair, not memorization of a
# specific sum. Chosen to spread across small/medium/large sums.
ALL_ARITH_PAIRS = [(a, b) for a in range(5) for b in range(5)]
ARITH_HELD_OUT_PAIRS = [(0, 4), (1, 3), (2, 2), (3, 4), (4, 4)]
ARITH_TRAIN_PAIRS = [p for p in ALL_ARITH_PAIRS if p not in ARITH_HELD_OUT_PAIRS]


def generate_arithmetic_episode(rng: random.Random, split: str = "train") -> dict:
    """School-0, arithmetic: "{a} plus {b} equals" -> the sum, as a
    word already in NUMBERS (zero..eight covers every possible sum of
    two operands in 0-4). `split="test"` forces (a, b) to be one of the
    5 pairs never seen as a training example -- a real held-out
    COMBINATION test, directly analogous to L3's held-out (size, color)
    pairs, now applied to whether addition generalizes to unseen
    operand pairs rather than being memorized per-pair."""
    pool = ARITH_TRAIN_PAIRS if split == "train" else ARITH_HELD_OUT_PAIRS
    a, b = rng.choice(pool)
    total = a + b
    instruction = f"{NUMBERS[a]} plus {NUMBERS[b]} equals"
    return {"a": a, "b": b, "sum": total, "sum_idx": total, "instruction": instruction}


def generate_rule_episode(rng: random.Random) -> dict:
    """School-0, logic/causal rules: teaches a GENERAL conditional rule
    ("if an object is {color} then it is {size}") -- quantified over
    ALL objects of that color, not a fact about one specific object --
    then asks a question that can only be answered by APPLYING the
    rule to a new instance identified by the rule's own premise color.
    This is a real deduction test (rule + observation -> conclusion,
    modus ponens), distinct from L5's fact recall: the answer is never
    stated anywhere, it must be derived."""
    premise_color = rng.choice(COLORS)
    conclusion_size = rng.choice(SIZES)
    rule = f"if an object is {premise_color} then it is {conclusion_size}"
    question = f"what size is the {premise_color} object"
    return {"rule": rule, "question": question, "premise_color": premise_color,
            "conclusion_size": conclusion_size, "answer_idx": SIZES.index(conclusion_size)}


def generate_cs_program_episode(rng: random.Random) -> dict:
    """School-0, Computer Science: "program execution" -- a genuinely
    different skill from raw arithmetic (`generate_arithmetic_episode`
    states both operands directly in one instruction). Here, two
    variable assignments ("x is {a}", "y is {b}") must first be tracked
    as a real symbol table -- TWO simultaneous bindings, directly
    connecting to this session's own L5 memory-stress finding that 2
    simultaneous facts sit right at the edge of what S can reliably
    hold -- before the values can be substituted and added. Real,
    disclosed test: does correct execution require MORE than the
    2-fact edge this project's own persistent memory already struggles
    with, or does composing retrieval with arithmetic change the
    picture?"""
    a, b = rng.randrange(5), rng.randrange(5)
    program = [f"x is {NUMBERS[a]}", f"y is {NUMBERS[b]}"]
    question = "what is x plus y"
    total = a + b
    return {"program": program, "question": question, "x": a, "y": b,
            "sum": total, "sum_idx": total}


def generate_physics_episode(rng: random.Random) -> dict:
    """School-0, Physics (plan section 8.2/9, first slice): a
    comparative-magnitude rule ("a large object needs more force than a
    small object") must be applied to two specific, per-episode objects
    identified only by color. Genuinely different skill from
    `generate_rule_episode`'s single-object classification: the answer
    is which of TWO named entities the rule picks out, not the
    conclusion for one premise -- a real magnitude-comparison /
    relational-inference test, not fact recall or single-instance
    deduction. Question-order (large-first vs small-first) is
    randomized so the model cannot shortcut on position."""
    large_color, small_color = rng.sample(COLORS, 2)
    teach = "a large object needs more force than a small object"
    scenario = f"the {large_color} object is large and the {small_color} object is small"
    first, second = (large_color, small_color) if rng.random() < 0.5 else (small_color, large_color)
    question = f"which object needs more force the {first} object or the {second} object"
    return {"teach": teach, "scenario": scenario, "question": question,
            "large_color": large_color, "small_color": small_color,
            "answer_color": large_color, "answer_idx": COLORS.index(large_color)}


PHYSICS_IDENTITY_LABELS = ["x", "y"]


def generate_physics_fixed_identity_episode(rng: random.Random) -> dict:
    """Real, decisive ablation on `generate_physics_episode`'s
    plateau (plan Phase 9, held-out acc stuck ~0.45-0.56, right at the
    "guess one of the two named colors" chance floor). Identical rule
    and structure, but the two entities are named by FIXED, reused
    symbol tokens (`x`/`y` -- literally the same tokens the CS program-
    execution task used to reach 97-100%) instead of a per-episode-
    varying COLOR. Isolates whether the reasoning rule itself is the
    bottleneck (predicts this stays ~50%) or whether COLOR's dynamic,
    per-episode entity identity was the real problem (predicts this
    jumps toward CS's ~100%) -- x/y's assignment to large/small is
    still randomized per episode so the answer can't be shortcut by
    always predicting one fixed symbol."""
    large_id, small_id = (PHYSICS_IDENTITY_LABELS if rng.random() < 0.5
                           else list(reversed(PHYSICS_IDENTITY_LABELS)))
    teach = "a large object needs more force than a small object"
    scenario = f"the {large_id} object is large and the {small_id} object is small"
    first, second = (large_id, small_id) if rng.random() < 0.5 else (small_id, large_id)
    question = f"which object needs more force the {first} object or the {second} object"
    return {"teach": teach, "scenario": scenario, "question": question,
            "large_id": large_id, "small_id": small_id,
            "answer_id": large_id, "answer_idx": PHYSICS_IDENTITY_LABELS.index(large_id)}


def generate_value_retrieval_episode(rng: random.Random) -> dict:
    """Real 2x2 diagnostic, cell 1 (plan Phase 9's entity-selection
    follow-up): identical program structure to
    `generate_cs_program_episode` (two bindings, "x is {a}", "y is
    {b}"), but the question asks for ONE bound VALUE directly ("what is
    x") instead of composing both via addition. Control cell: if this
    is easy (as expected, given CS's own 97-100%), it confirms single-
    fact value retrieval from a 2-fact program isn't the bottleneck --
    isolating composition vs. retrieval before testing retrieval of an
    ENTITY REFERENCE instead of a value (`generate_entity_select_episode`)."""
    a, b = rng.randrange(5), rng.randrange(5)
    program = [f"x is {NUMBERS[a]}", f"y is {NUMBERS[b]}"]
    ask_x = rng.random() < 0.5
    question = "what is x" if ask_x else "what is y"
    answer = a if ask_x else b
    return {"program": program, "question": question, "x": a, "y": b,
            "answer": answer, "answer_idx": answer}


def generate_entity_select_episode(rng: random.Random) -> dict:
    """Real 2x2 diagnostic, cell 2 (plan Phase 9's entity-selection
    follow-up) -- the Physics ablation's real successor. Strips away
    EVERYTHING Physics had beyond the core operation: no arithmetic, no
    physical rule, no comparison, no colors. Just "x is a widget, y is
    a gadget, which is the widget" -- a pure (entity -> property) bind
    for x and y, then a question that must SELECT AND OUTPUT the
    entity (x or y) satisfying a named property, not compute a derived
    value. Property-to-symbol assignment and which property is asked
    about are both randomized per episode so neither can be shortcut.
    Directly comparable to `generate_value_retrieval_episode`: same
    program shape, same output space size ({x, y} vs one NUMBERS digit
    -- both 2-ish-way after conditioning), the only real difference is
    whether the answer is a VALUE or a REFERENCE to an entity."""
    prop_x, prop_y = (ENTITY_WORDS if rng.random() < 0.5 else list(reversed(ENTITY_WORDS)))
    program = [f"x is a {prop_x}", f"y is a {prop_y}"]
    asked_prop = rng.choice(ENTITY_WORDS)
    question = f"which is the {asked_prop}"
    answer_id = "x" if prop_x == asked_prop else "y"
    return {"program": program, "question": question, "prop_x": prop_x, "prop_y": prop_y,
            "asked_prop": asked_prop, "answer_id": answer_id,
            "answer_idx": PHYSICS_IDENTITY_LABELS.index(answer_id)}
