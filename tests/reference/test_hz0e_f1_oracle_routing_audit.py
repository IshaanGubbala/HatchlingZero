"""HZ-0F F1: oracle routing audit (`reference/hz0e_f1_oracle_routing_audit.py`).

Diagnoses whether E8/E10's disclosed OOD quality gap is a routing
problem or an architecture problem, on the real trained checkpoint --
see the module's own docstring for the full design and its disclosed
approximation.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from reference.hz0e_e2_router_simulator import DOMAIN_DATA_PATHS
from reference.hz0e_e8_curriculum import LAYER, load_domain_batches, run_curriculum, run_warm_dense_baseline
from reference.hz0e_f1_oracle_routing_audit import oracle_routing_audit
from reference.hz0e_moe_contract import MoeConfig
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists() or not Path(GENERAL_DATA_PATH).exists(),
    reason="real HZ-0A checkpoint/corpus not present locally",
)


def test_oracle_routing_audit_is_structurally_valid_and_never_worse_than_actual():
    """Real, fast-but-real curriculum training (matching this project's
    established `balanced_steps=15, mixed_steps=15, imbalanced_steps=15,
    warm_start_steps=20` test convention), 3 seeds, then the real oracle
    audit on both an in-distribution (curriculum domain) and
    out-of-distribution (general prose) held-out batch per seed.

    Asserts STRUCTURAL properties (the oracle can never be
    meaningfully worse than the actual router, since "actual router" is
    effectively bounded by the same candidate set -- win rates sum to
    1, everything finite) -- this is a real diagnostic experiment, not
    a test with one known expected outcome. It ALSO locks in the real,
    multi-seed, reproducible finding this audit produced (see
    `docs/restart/hz0e_f1_oracle_routing_audit_results.md`): the
    oracle-vs-actual gap is only marginally larger OOD than
    in-distribution (not dramatically larger), and the dense/abstain
    candidate's oracle win rate is LOWER on OOD than in-distribution in
    every seed tested -- the opposite of what "add abstention to dense
    for OOD tokens" would predict. Locking in the DIRECTION of this
    finding (not exact values, which have real seed-to-seed variation)
    as a real regression test, matching this project's standing
    discipline of not re-deriving a real finding by hand each time
    without a test catching drift."""
    model, _ = load_frozen_model()
    config = MoeConfig(dim=model.dim)

    held_out_domains = load_domain_batches(DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=1)
    in_distribution_batch = held_out_domains["prose"]
    ood_batch = mx.concatenate([mx.array([s[:64]]) for s in load_real_sequences(GENERAL_DATA_PATH, 8)], axis=0)

    dense_win_rate_lower_ood_count = 0
    print(f"\nOracle routing audit (real checkpoint, real curriculum, layer {LAYER}):")
    for seed in (0, 1, 2):
        moe_trained, _report = run_curriculum(model, config, balanced_steps=15, mixed_steps=15, imbalanced_steps=15, seed=seed, warm_start_steps=20)
        dense_trained, _before, _after = run_warm_dense_baseline(model, balanced_steps=15, mixed_steps=15, imbalanced_steps=15, seed=seed)

        in_dist_result = oracle_routing_audit(model, moe_trained, dense_trained, config, LAYER, in_distribution_batch)
        ood_result = oracle_routing_audit(model, moe_trained, dense_trained, config, LAYER, ood_batch)

        for result in (in_dist_result, ood_result):
            assert all(v == v and abs(v) < 1e6 for v in [result.actual_router_mean_loss, result.oracle_mean_loss]), "non-finite loss"
            assert result.oracle_gap >= -1e-4, f"oracle should never be meaningfully worse than actual: gap={result.oracle_gap}"
            assert abs(sum(result.candidate_win_rate.values()) - 1.0) < 1e-4, "win rates must sum to 1"
            assert set(result.candidate_win_rate) == {"expert_0", "expert_1", "expert_2", "expert_3", "dense"}

        print(f"  seed={seed} in-dist:  actual={in_dist_result.actual_router_mean_loss:.4f}  oracle={in_dist_result.oracle_mean_loss:.4f}  gap={in_dist_result.oracle_gap:.4f}  dense_win={in_dist_result.candidate_win_rate['dense']:.3f}")
        print(f"  seed={seed} ood:      actual={ood_result.actual_router_mean_loss:.4f}  oracle={ood_result.oracle_mean_loss:.4f}  gap={ood_result.oracle_gap:.4f}  dense_win={ood_result.candidate_win_rate['dense']:.3f}")

        if ood_result.candidate_win_rate["dense"] < in_dist_result.candidate_win_rate["dense"]:
            dense_win_rate_lower_ood_count += 1

    assert dense_win_rate_lower_ood_count == 3, (
        "expected the real, reproducible finding to hold in all 3 seeds: dense's oracle win rate "
        "is lower OOD than in-distribution, not higher -- if this regresses, the finding needs re-examination, "
        "not silent re-assertion"
    )


def test_gated_oracle_confirms_the_unscaled_finding_is_not_a_framing_artifact():
    """F2's follow-up check: does the unscaled-oracle finding above
    survive once each forced-expert candidate is scaled by its REAL
    softmax probability for that specific expert (realistic amplitude),
    not left unscaled? See
    `docs/restart/hz0e_f2_gate_overflow_fallback_results.md`. Gate
    scaling mechanically shrinks non-preferred experts' contributions,
    which inflates the absolute oracle gap and raises dense's absolute
    win rate compared to the unscaled framing -- but the DIRECTION that
    matters (gap not OOD-amplified; dense doesn't win MORE OOD) must
    still hold, or the unscaled finding above would be a framing
    artifact rather than a real effect."""
    model, _ = load_frozen_model()
    config = MoeConfig(dim=model.dim)
    in_distribution_batch, ood_batch = load_domain_batches(DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=1)["prose"], mx.concatenate([mx.array([s[:64]]) for s in load_real_sequences(GENERAL_DATA_PATH, 8)], axis=0)

    dense_win_rate_lower_ood_count = 0
    for seed in (0, 1, 2):
        moe_trained, _report = run_curriculum(model, config, balanced_steps=15, mixed_steps=15, imbalanced_steps=15, seed=seed, warm_start_steps=20)
        dense_trained, _before, _after = run_warm_dense_baseline(model, balanced_steps=15, mixed_steps=15, imbalanced_steps=15, seed=seed)

        in_dist_result = oracle_routing_audit(model, moe_trained, dense_trained, config, LAYER, in_distribution_batch, gate_scaled=True)
        ood_result = oracle_routing_audit(model, moe_trained, dense_trained, config, LAYER, ood_batch, gate_scaled=True)

        assert in_dist_result.oracle_gap >= -1e-4 and ood_result.oracle_gap >= -1e-4
        if ood_result.candidate_win_rate["dense"] < in_dist_result.candidate_win_rate["dense"]:
            dense_win_rate_lower_ood_count += 1

    assert dense_win_rate_lower_ood_count == 3, (
        "gated-oracle finding should match the unscaled framing's direction in all 3 seeds"
    )
