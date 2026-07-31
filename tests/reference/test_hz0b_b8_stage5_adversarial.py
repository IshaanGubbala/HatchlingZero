"""HZ-0B B8 Stage 5 tests: the plan's own 7 named adversarial scenarios,
each checked against a real, concrete pass/fail criterion -- not just
"ran without crashing." Fast and deterministic (no LM, no training run):
these are properties of the memory mechanism itself
(`reference/hz0b_b8_stage5_adversarial.py`), the same testing level B2/B3
used.
"""
import mlx.core as mx

from reference.hz0b_b8_stage5_adversarial import (
    scenario_capacity_pressure,
    scenario_contradictory_later_information,
    scenario_distractors,
    scenario_malicious_overwrite_attempt,
    scenario_near_identical_keys,
    scenario_reset_boundaries,
    scenario_stale_memories,
    scenario_stale_vs_fresh_competition,
)


def cosine(a, b) -> float:
    return float(mx.sum(a * b) / (mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b * b)) + 1e-8))


def test_contradictory_later_information_resolves_to_the_newer_fact():
    result = scenario_contradictory_later_information()
    assert cosine(result["readout"][0], result["fact2"][0]) > 0.99
    assert cosine(result["readout"][0], result["fact1"][0]) < 0.5


def test_distractors_do_not_disturb_the_real_fact():
    result = scenario_distractors()
    assert cosine(result["readout"][0], result["real_value"][0]) > 0.99


def test_malicious_overwrite_attempt_is_rejected():
    result = scenario_malicious_overwrite_attempt()
    assert bool(result["rejected"][0]) is True
    assert cosine(result["readout"][0], result["legit_value"][0]) > 0.99
    assert cosine(result["readout"][0], result["attacker_value"][0]) < 0.5


def test_near_identical_keys_are_kept_as_distinct_memories_fixed_2026_07_30():
    """Was a real, disclosed vulnerability (see git history /
    docs/restart/hz0b_b8_stage5_results.md): two GENUINELY DIFFERENT
    facts at cosine 0.995 used to be silently conflated into one slot,
    because B1/B2's match threshold (0.95) treated "highly correlated"
    as "the same fact, update it." Fixed by raising the threshold to
    0.999 (near-EXACT key identity required for the in-place-update
    path; `test_overwrite_existing_fact` -- literal same key -- is
    unaffected) -- now confirmed to route to two distinct, independently
    correct slots."""
    result = scenario_near_identical_keys()
    assert result["similarity"] > 0.9, "test setup sanity: keys must actually be near-identical, not just similar"
    assert not bool(mx.array_equal(result["slot_a"], result["slot_b"])), "near-identical-but-different keys must now route to distinct slots"
    assert cosine(result["readout_a"][0], result["value_a"][0]) > 0.99, "fact A survives, undisturbed"
    assert cosine(result["readout_b"][0], result["value_b"][0]) > 0.99, "fact B survives, undisturbed"


def test_stale_memory_confidence_genuinely_decays():
    result = scenario_stale_memories(decay_steps=20, decay_rate=0.9)
    assert result["decayed_confidence"] < result["initial_confidence"] * 0.2, "20 steps at decay_rate=0.9 should reduce confidence by more than 5x"


def test_stale_memory_alone_still_retrieves_fine_no_competition_to_lose_to():
    """When a stale memory is the ONLY match for a query, it still
    retrieves fully -- correct: confidence-weighting should only matter
    RELATIVE to a competing alternative, not zero out a memory that
    happens to be the sole candidate. See the competition test below for
    where confidence-weighting actually changes the outcome."""
    result = scenario_stale_memories(decay_steps=50, decay_rate=0.9)
    assert result["decayed_confidence"] < 0.01
    assert cosine(result["readout_hard"][0], result["value"][0]) > 0.99


def test_confidence_weighted_read_prefers_fresh_over_stale_when_competing_fixed_2026_07_30():
    """Was a real, disclosed gap (docs/restart/hz0b_b8_stage5_results.md
    finding 2): confidence had zero effect on read strength, only on
    write-eviction scoring. Fixed by adding `confidence_weighted=True`
    (now the default) to `read()` -- log(confidence) biases the
    similarity score, so a fresh, high-confidence memory now correctly
    wins a tie-similarity competition against a heavily-decayed one."""
    result = scenario_stale_vs_fresh_competition(decay_steps=50, decay_rate=0.9)
    assert result["stale_confidence"] < 0.01
    assert result["fresh_confidence"] > 0.9
    assert cosine(result["readout_weighted"][0], result["fresh_value"][0]) > 0.99, "confidence-weighted read must prefer the fresh memory"
    assert cosine(result["readout_weighted"][0], result["stale_value"][0]) < 0.5
    # Also confirms this genuinely IS what confidence_weighted=True changes
    # -- the unweighted path is a same-similarity tie, won by whichever
    # slot argmax picks first (lower index), not by freshness.
    assert cosine(result["readout_unweighted"][0], result["stale_value"][0]) > 0.99, "unweighted read has no basis to prefer either slot -- ties go to the lower index (the stale one, written first), demonstrating what confidence_weighted actually fixes"


def test_capacity_pressure_protected_memory_survives_and_no_crash():
    result = scenario_capacity_pressure(num_facts=12)
    assert cosine(result["readout"][0], result["protected_value"][0]) > 0.99, "protected memory must survive capacity pressure from 11 competing writes"
    assert result["occupied_slots"] <= 8  # NUM_SLOTS -- no silent overflow/corruption
    assert result["rejected_count"] == 0, "unprotected slots should absorb the pressure via eviction, not force protected-slot rejections in this scenario (only one slot is protected)"


def test_reset_wipes_everything_including_protected_memories():
    result = scenario_reset_boundaries()
    assert result["confidence_after_reset"] == 0.0
    assert result["protection_after_reset"] == 0.0, "reset is not gated by protection -- a protected memory does not survive reset, matching B1's 'full zero, matches legacy semantics exactly'"
    assert cosine(result["readout_after_reset"][0], result["value"][0]) < 0.5, "no leakage of the pre-reset value across the reset boundary"
