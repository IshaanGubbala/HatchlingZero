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


def test_near_identical_keys_are_silently_conflated_a_real_adversarial_vulnerability():
    """Real, disclosed finding, not the hoped-for "keeps them distinct"
    result: two GENUINELY DIFFERENT facts, keyed with cosine 0.995
    (above B1/B2's 0.95 match threshold), get silently merged into ONE
    slot -- key_b's write is treated as an UPDATE to key_a's existing
    entry, not a new, distinct memory. Confirmed directly (both writes
    land in slot 0, not two different slots). This is different from the
    `malicious_overwrite_attempt` scenario (which tests an explicit,
    same-slot overwrite of a PROTECTED memory, and correctly fails) --
    this is an UNPROTECTED, accidental-or-adversarial conflation that
    happens purely because of key geometry, with no protection mechanism
    guarding against it. A real, working-as-designed consequence of B1
    decision 8's fixed 0.95 similarity threshold, not a bug in this test
    or the simulator -- but a genuine adversarial surface: an attacker
    (or just unlucky embedding collisions) who can craft a key close
    enough to an existing one can silently overwrite it without
    triggering any protection check at all, since the write path treats
    it as a legitimate update to the SAME fact, not a competing write
    that protection logic would evaluate."""
    result = scenario_near_identical_keys()
    assert result["similarity"] > 0.9, "test setup sanity: keys must actually be near-identical, not just similar"
    assert bool(mx.array_equal(result["slot_a"], result["slot_b"])), "both writes land in the same slot -- this IS the vulnerability, not an artifact"
    # value_b's write overwrote value_a's -- reading either key now
    # returns value_b, not value_a. This is the concerning behavior,
    # documented directly rather than asserted away.
    assert cosine(result["readout_a"][0], result["value_b"][0]) > 0.9, "confirms the conflation: querying key_a now retrieves fact B's value, not fact A's"
    assert cosine(result["readout_a"][0], result["value_a"][0]) < 0.5, "fact A's own value is gone -- silently, with no protection check ever triggered"


def test_stale_memory_confidence_genuinely_decays():
    result = scenario_stale_memories(decay_steps=20, decay_rate=0.9)
    assert result["decayed_confidence"] < result["initial_confidence"] * 0.2, "20 steps at decay_rate=0.9 should reduce confidence by more than 5x"


def test_stale_memory_hard_read_is_unaffected_by_confidence_a_real_disclosed_gap():
    """Honest finding, not assumed: B2's read() does not weight by
    confidence at all -- a stale-but-key-matching memory still retrieves
    at FULL strength under a hard (top-1) read. This is disclosed as a
    real property of the current design, not silently treated as "staleness
    naturally protects against retrieving outdated content" (it does not,
    on its own -- only decay of the underlying VALUE toward zero, or
    eviction, would)."""
    result = scenario_stale_memories(decay_steps=50, decay_rate=0.9)  # confidence now extremely low
    assert result["decayed_confidence"] < 0.01
    assert cosine(result["readout_hard"][0], result["value"][0]) > 0.99, "a hard read still fully retrieves a near-zero-confidence memory -- confidence does not gate retrieval strength in this design"


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
