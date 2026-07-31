"""HZ-0B Phase B8, Stage 4 ("Natural sequences"): the plan's own 7 named
patterns -- multi-turn conversations, evolving constraints, tool results
needed later, variable assignments, code symbols, document facts,
changing user preferences within a session -- composed into ONE realistic
session (not 7 isolated micro-tests, the way Stage 5's adversarial probes
were) and tested against B2's memory simulator.

Deliberately different framing from Stage 5
(`reference/hz0b_b8_stage5_adversarial.py`): Stage 5 asks "does the
mechanism resist attacks/edge cases"; Stage 4 asks "does the mechanism
correctly serve normal, expected, composite usage" -- these should mostly
PASS, since nothing here is adversarial, just realistic and layered (many
different memory items alive at once, updated at different times, queried
after realistic delays, with ordinary conversational noise in between).
"""
from __future__ import annotations

import mlx.core as mx

from reference.hz0b_memory_simulator import protect, read, reset, write

NUM_SLOTS, KEY_DIM, VALUE_DIM = 16, 24, 24
# 6 named items + up to a dozen distractor turns need to fit without
# capacity-driven eviction -- capacity PRESSURE is Stage 5's job
# (reference/hz0b_b8_stage5_adversarial.py's own dedicated scenario),
# not this one; too few slots here would make ordinary session noise
# accidentally evict real facts, which isn't what "natural sequences"
# is testing.


def _onehot(dim: int, index: int, batch: int = 1) -> mx.array:
    row = [1.0 if i == index else 0.0 for i in range(dim)]
    return mx.array([row for _ in range(batch)])


# Named "item" slots used across the composite session -- each represents
# one of B8 Stage 4's 7 required patterns, given its own distinct
# key/value index range so they can coexist in one memory bank at once.
VARIABLE_X = 0            # "variable assignments" -- reassigned mid-session
CODE_SYMBOL_FN = 1         # "code symbols" -- a function/class name -> its definition location, referenced much later
CONSTRAINT = 2             # "evolving constraints" -- a rule that gets tightened mid-session
TOOL_RESULT = 3            # "tool results needed later" -- written once, needed near the end
DOCUMENT_FACT = 4          # "document facts" -- read from a document early, referenced at the end
USER_PREFERENCE = 5        # "changing user preferences" -- set once, changed once, must reflect the latest

DISTRACTOR_BASE = 6        # ordinary conversational turns with nothing worth remembering


def run_composite_natural_session(*, num_distractor_turns: int = 4) -> dict:
    """One realistic multi-turn session: writes and reads for all 6 named
    item types above, interleaved with ordinary distractor turns, in a
    plausible temporal order (early setup, mid-session updates, a
    late-session battery of queries -- the way a real assistant session
    would actually unfold). Returns the final read for each item plus
    what it SHOULD be, for the test suite to check directly."""
    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    step = 0

    def do_write(key_index, value_index, *, protect_it=False):
        nonlocal state, step
        key, value = _onehot(KEY_DIM, key_index), _onehot(VALUE_DIM, value_index) * 5.0
        state, slot, _ = write(state, key, value, mx.array([1.0]), step=step)
        if protect_it:
            state = protect(state, slot, mx.array([1.0]))
        step += 1
        return value

    # Turn 1-2: early setup -- document fact read once, code symbol noted.
    document_fact_value = do_write(DOCUMENT_FACT, 0)
    code_symbol_value = do_write(CODE_SYMBOL_FN, 1)

    # Turn 3: variable assignment.
    do_write(VARIABLE_X, 2)  # x = <value 2>, will be reassigned later

    # Turn 4: initial constraint, explicitly protected (a real system
    # constraint shouldn't be casually overwritten by ordinary chatter).
    do_write(CONSTRAINT, 3, protect_it=True)

    # Turns 5-8: ordinary conversational noise, nothing worth remembering.
    # Cycles through a SMALL fixed set of "off-topic" keys regardless of
    # how many distractor turns there are -- a real long conversation
    # keeps circling back to a few kinds of small talk, it doesn't mint a
    # brand new distinct memory-worthy topic every single turn. Capacity
    # PRESSURE from many genuinely distinct items is Stage 5's own
    # dedicated scenario, not this one.
    num_distractor_keys = 2
    for i in range(num_distractor_turns):
        do_write(DISTRACTOR_BASE + (i % num_distractor_keys), 6 + (i % 3))

    # Turn 9: user sets an initial preference.
    do_write(USER_PREFERENCE, 7)

    # Turn 10: tool call result -- written once, needed much later.
    tool_result_value = do_write(TOOL_RESULT, 8)

    # More distractor turns (the "needed much later" gap) -- same small
    # fixed set of off-topic keys, not new ones each time.
    for i in range(num_distractor_turns):
        do_write(DISTRACTOR_BASE + (i % num_distractor_keys), 9 + (i % 3))

    # Turn N: variable reassignment -- "x = <new value>", must now read as the NEW value.
    variable_x_final_value = do_write(VARIABLE_X, 10)

    # Turn N+1: user changes their preference -- must now read as the NEW preference.
    user_preference_final_value = do_write(USER_PREFERENCE, 11)

    # A real evolving constraint: the ORIGINAL constraint was protected on
    # purpose (system rules shouldn't casually change) -- attempting to
    # "evolve" it via an ordinary write should be REFUSED, distinguishing
    # this from a legitimate explicit update (which would go through
    # `update()`, not a competing `write()` -- not exercised here, this
    # composite session intentionally tests that casual mid-session writes
    # cannot silently override a protected system constraint).
    # No slot_idx here -- auto-routing detects the SAME key (similarity
    # ~1.0 to whatever slot CONSTRAINT actually landed in, which this
    # function never explicitly tracked) as a match and attempts an
    # in-place update, which protection then blocks.
    key_constraint = _onehot(KEY_DIM, CONSTRAINT)
    attempted_new_constraint = _onehot(VALUE_DIM, 12) * 5.0
    state, _, constraint_change_rejected = write(state, key_constraint, attempted_new_constraint, mx.array([1.0]), step=step)
    step += 1

    # Final battery of reads, as if the session is wrapping up and the
    # assistant needs to recall everything at once.
    document_fact_readout, _ = read(state, _onehot(KEY_DIM, DOCUMENT_FACT), hard=True)
    code_symbol_readout, _ = read(state, _onehot(KEY_DIM, CODE_SYMBOL_FN), hard=True)
    variable_x_readout, _ = read(state, _onehot(KEY_DIM, VARIABLE_X), hard=True)
    constraint_readout, _ = read(state, key_constraint, hard=True)
    user_preference_readout, _ = read(state, _onehot(KEY_DIM, USER_PREFERENCE), hard=True)
    tool_result_readout, _ = read(state, _onehot(KEY_DIM, TOOL_RESULT), hard=True)

    return {
        "document_fact_readout": document_fact_readout, "document_fact_expected": document_fact_value,
        "code_symbol_readout": code_symbol_readout, "code_symbol_expected": code_symbol_value,
        "variable_x_readout": variable_x_readout, "variable_x_expected": variable_x_final_value,
        "constraint_readout": constraint_readout, "constraint_expected_unchanged_value": _onehot(VALUE_DIM, 3) * 5.0,
        "constraint_change_rejected": constraint_change_rejected,
        "user_preference_readout": user_preference_readout, "user_preference_expected": user_preference_final_value,
        "tool_result_readout": tool_result_readout, "tool_result_expected": tool_result_value,
    }
