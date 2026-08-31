"""Genuinely sequential reasoning task, 2026-08-31 -- replaces the
entity-location chain task for the progressive-latentization curriculum
specifically (plans/newnewplan.md's "Progressive Latentization Training"
proposal). Real, important difference from
scripts/hz0h_bdh_shortcut_resistant_chain_task.py: that task is a graph
LOOKUP chain (entity_i -> entity_{i+1} -> ... -> location) -- correctness
doesn't depend on reading the hop sentences in any particular order,
which is exactly why it could be shuffled to kill the positional
shortcut. A register machine is different: `r=3; add 2; mul 2` genuinely
requires applying the operations IN ORDER (r=3 -> 5 -> 10, not any other
order), so shuffling the real operation clauses would change the
correct answer, not just remove a shortcut. This task keeps the real
operation clauses in true chronological order and instead defeats
positional shortcuts with DECOY clauses about unrelated variable names,
inserted at random positions -- a model that just reads "the last
clause in the text" gets the decoy's value, not r's, unless it actually
tracks which variable is being asked about and threads its true value
through every real step in order.

Register values are kept in Z/10Z (mod 10) throughout, so every
intermediate value is a single decimal digit -- deliberately so a
per-step supervision signal can reuse the model's own real byte-level
lm_head directly (predict the ASCII digit '0'-'9' after each real
operation) rather than needing a separate classification head, per the
LOTUS-style step-alignment idea: p_i = LMHead(h_i), target = the ASCII
byte of r's true value after step i. Final answer is also just that
same digit, read after the full chain.
"""
from __future__ import annotations

import random

VAR_NAMES = ["r", "s", "t", "u", "v", "w", "x", "y", "z", "p", "q", "m", "n", "k", "j", "i"]
OPS = ["add", "sub", "mul"]


def _apply(op: str, val: int, k: int) -> int:
    if op == "add":
        return (val + k) % 10
    if op == "sub":
        return (val - k) % 10
    if op == "mul":
        return (val * k) % 10
    raise ValueError(op)


def generate_register_machine_example(rng: random.Random, n_steps: int, n_decoys: int | None = None):
    """Returns (text, step_targets, final_answer, decoy_answer).

    step_targets: list of n_steps ints, r's true value after each REAL
    operation, in true chronological order -- for LOTUS-style per-round
    supervision (only used by Arm D / the progressive-latentization
    curriculum, not by ordinary LM training).
    final_answer / decoy_answer: single digits 0-9, matching the
    original task's (correct_idx, shortcut_idx) convention -- decoy_answer
    is the value of whichever decoy variable's clause happens to be
    LAST in the shuffled text, i.e. what a pure "read the last clause"
    shortcut would output.
    """
    if n_decoys is None:
        n_decoys = max(1, n_steps // 2)
    names = rng.sample(VAR_NAMES, 1 + n_decoys)
    target_var = names[0]
    decoy_vars = names[1:]

    start_val = rng.randint(0, 9)
    real_clauses = [f"{target_var}={start_val}."]
    val = start_val
    step_targets = []
    for _ in range(n_steps):
        op = rng.choice(OPS)
        k = rng.randint(1, 4)
        val = _apply(op, val, k)
        real_clauses.append(f"{op} {k}.")
        step_targets.append(val)
    final_answer = val

    decoy_clauses = []
    decoy_final_per_var = {}
    for dv in decoy_vars:
        dval = rng.randint(0, 9)
        n_dsteps = rng.randint(1, max(1, n_steps))
        clause_group = [f"{dv}={dval}."]
        for _ in range(n_dsteps):
            op = rng.choice(OPS)
            k = rng.randint(1, 4)
            dval = _apply(op, val=dval, k=k)
            clause_group.append(f"{op} {k}.")
        decoy_final_per_var[dv] = dval
        decoy_clauses.append(clause_group)

    # Real clauses (all of them, including r's start value) keep their
    # relative order -- only decoy GROUPS get inserted at random points
    # around them, and each decoy group's OWN internal clause order is
    # preserved (it's also a real, order-dependent mini-chain, just for
    # a variable never asked about).
    segments = [real_clauses] + decoy_clauses
    rng.shuffle(segments)
    # after shuffling groups, merge -- but to make "read the last clause"
    # a real, checkable shortcut, further interleave at the clause level
    # only across group BOUNDARIES (never breaking a group's internal
    # order): flatten segments in the shuffled group order.
    all_clauses = [c for group in segments for c in group]

    text = " ".join(all_clauses) + f" What is {target_var}?"
    last_clause_group = segments[-1]
    if last_clause_group is real_clauses:
        decoy_answer = final_answer  # last clause belongs to the real chain -- no shortcut available this example
    else:
        # which decoy var owns the last group
        last_var = last_clause_group[0].split("=")[0]
        decoy_answer = decoy_final_per_var[last_var]

    return text, step_targets, final_answer, decoy_answer


def generate_register_machine_cot_example(rng: random.Random, n_steps: int) -> tuple[str, str, list[int], int]:
    """Clean explicit chain-of-thought trace, NO decoy noise, real
    operations always applied in true order -- for Arm C (explicit CoT
    SFT) and Arm D (progressive latentization) training data. Deliberately
    NOT shortcut-resistant: shortcut-resistance is an EVALUATION concern
    (the compact generate_register_machine_example above, used for the
    shared R x step-count eval matrix across all 4 arms), not a training-
    data concern -- train on clean worked traces, evaluate transfer onto
    the harder adversarial format separately.

    Real structural fix vs an earlier draft of this function: the
    OPERATIONS themselves are never secret -- in any real CoT setup the
    question already states what to compute, and T_1..T_n are the
    INTERMEDIATE RESULTS of applying it, not the operations. So the
    question prefix states r's start value AND every operation up front
    (nothing to latentize there), and only the space-separated sequence
    of intermediate result digits after "Results:" is what progressively
    latentizes -- one digit token per real reasoning step, a clean,
    fixed-width span Arm D can hide from the front.

    Returns (question_prefix, results_and_answer_suffix, step_targets,
    final_answer) split at exactly the point where latentization begins,
    so callers don't have to re-parse the text.
    """
    var = rng.choice(VAR_NAMES)
    start_val = rng.randint(0, 9)
    val = start_val
    ops = []
    step_targets = []
    for _ in range(n_steps):
        op = rng.choice(OPS)
        k = rng.randint(1, 4)
        val = _apply(op, val, k)
        ops.append(f"{op} {k}")
        step_targets.append(val)
    question_prefix = f"{var}={start_val}. " + " ".join(ops) + ". Results:"
    results_suffix = "".join(f" {d}" for d in step_targets) + f". Answer: {val}"
    return question_prefix, results_suffix, step_targets, val


if __name__ == "__main__":
    rng = random.Random(0)
    for n_steps in [1, 2, 4]:
        text, steps, ans, decoy = generate_register_machine_example(rng, n_steps)
        print(f"n_steps={n_steps}: {text}\n  step_targets={steps} final_answer={ans} decoy_answer={decoy}\n")
    print("--- CoT variant ---")
    for n_steps in [1, 2, 4]:
        q, suffix, steps, ans = generate_register_machine_cot_example(rng, n_steps)
        print(f"n_steps={n_steps}: {q!r} + {suffix!r}\n  step_targets={steps} answer={ans}\n")
