"""HZ-0B B8 Stage 4 tests: one realistic composite session exercising all
6 named "item" patterns from the plan's Stage 4 list (multi-turn
conversation structure is the session itself; code symbols, variable
assignments, evolving/protected constraints, tool results needed later,
document facts, and changing user preferences are the specific items
tracked within it) plus 5-8 ordinary distractor turns. Everything here is
expected to PASS -- unlike Stage 5, nothing is adversarial.
"""
import mlx.core as mx

from reference.hz0b_b8_stage4_natural_sequences import run_composite_natural_session


def cosine(a, b) -> float:
    return float(mx.sum(a * b) / (mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b * b)) + 1e-8))


def test_document_fact_survives_the_whole_session():
    result = run_composite_natural_session()
    assert cosine(result["document_fact_readout"][0], result["document_fact_expected"][0]) > 0.99


def test_code_symbol_survives_the_whole_session():
    result = run_composite_natural_session()
    assert cosine(result["code_symbol_readout"][0], result["code_symbol_expected"][0]) > 0.99


def test_variable_reassignment_reads_as_the_latest_value_not_the_original():
    result = run_composite_natural_session()
    assert cosine(result["variable_x_readout"][0], result["variable_x_expected"][0]) > 0.99


def test_protected_constraint_resists_a_casual_mid_session_overwrite():
    result = run_composite_natural_session()
    assert bool(result["constraint_change_rejected"][0]) is True
    assert cosine(result["constraint_readout"][0], result["constraint_expected_unchanged_value"][0]) > 0.99


def test_tool_result_is_recalled_correctly_after_a_long_gap():
    result = run_composite_natural_session()
    assert cosine(result["tool_result_readout"][0], result["tool_result_expected"][0]) > 0.99


def test_changed_user_preference_reads_as_the_new_one_not_the_original():
    result = run_composite_natural_session()
    assert cosine(result["user_preference_readout"][0], result["user_preference_expected"][0]) > 0.99


def test_session_holds_up_with_heavier_distractor_load():
    """Same session, more ordinary conversational noise -- everything
    above should still hold with a longer, busier session (a real
    multi-turn conversation has many more irrelevant turns than
    memory-worthy ones)."""
    result = run_composite_natural_session(num_distractor_turns=12)
    assert cosine(result["document_fact_readout"][0], result["document_fact_expected"][0]) > 0.99
    assert cosine(result["variable_x_readout"][0], result["variable_x_expected"][0]) > 0.99
    assert cosine(result["tool_result_readout"][0], result["tool_result_expected"][0]) > 0.99
    assert cosine(result["user_preference_readout"][0], result["user_preference_expected"][0]) > 0.99
    assert bool(result["constraint_change_rejected"][0]) is True
