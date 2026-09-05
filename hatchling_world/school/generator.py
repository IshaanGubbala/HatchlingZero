"""Procedural example generators for School-0, plans/Hatchling world.md
section 8.2. Real, deterministic (seeded), closed-vocabulary generation,
same discipline as hatchling_world.language.nursery_generator."""
from __future__ import annotations

import random

from hatchling_world.language.tokenizer import COLORS, NUMBERS, SIZES

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
