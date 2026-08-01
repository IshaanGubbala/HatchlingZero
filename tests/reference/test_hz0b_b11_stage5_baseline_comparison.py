"""HZ-0B B11: regression tests locking in the cross-baseline adversarial
scenario results (scripts/hz0b_b11_stage5_baseline_comparison.py,
docs/restart/hz0b_b11_stage5_baseline_results.md)."""
from __future__ import annotations

from scripts.hz0b_b11_stage5_baseline_comparison import (
    run_scenario_capacity_pressure,
    run_scenario_contradictory,
    run_scenario_distractors,
    run_scenario_near_identical_keys,
    run_scenario_noisy_query,
    run_scenario_noisy_query_with_distractors,
)


def test_hz0b_passes_contradictory_distractors_and_near_identical_keys():
    assert run_scenario_contradictory()["HZ-0B"] is True
    assert run_scenario_distractors()["HZ-0B"] is True
    assert run_scenario_near_identical_keys()["HZ-0B"] is True


def test_hz0b_correctly_evicts_unprotected_memory_under_real_capacity_pressure():
    """This is an intentional 'fail' -- correct eviction behavior when
    protection wasn't invoked, not a bug. Locking in the honest result,
    not a passing bar."""
    assert run_scenario_capacity_pressure()["HZ-0B"] is False


def test_no_memory_baseline_fails_everything_by_construction():
    assert run_scenario_contradictory()["no-memory"] is False
    assert run_scenario_distractors()["no-memory"] is False
    assert run_scenario_near_identical_keys()["no-memory"] is False


def test_scenario_5_confound_large_recurrent_and_long_context_pass_vacuously():
    """Documents the real confound: with only one item ever stored,
    content-blind/single-item baselines trivially 'pass' regardless of
    query quality."""
    result = run_scenario_noisy_query()
    assert result["large-recurrent"] is True
    assert result["long-context"] is True


def test_scenario_6_fixes_the_confound_both_now_correctly_fail():
    result = run_scenario_noisy_query_with_distractors()
    assert result["large-recurrent"] is False
    assert result["long-context"] is False
    assert result["HZ-0B"] is True
    assert result["external-retrieval"] is True


def test_simple_kv_cache_wins_exact_match_but_fails_noisy_query():
    assert run_scenario_contradictory()["simple-kv-cache"] is True
    assert run_scenario_noisy_query_with_distractors()["simple-kv-cache"] is False


def test_external_retrieval_fails_contradictory_info_no_update_semantics():
    """Structural gap: unbounded top-1 similarity search has no
    update/replace mechanism -- contradictory writes to the same key just
    accumulate, and argmax ties break toward the OLDER entry."""
    assert run_scenario_contradictory()["external-retrieval"] is False
    assert run_scenario_noisy_query_with_distractors()["external-retrieval"] is True
