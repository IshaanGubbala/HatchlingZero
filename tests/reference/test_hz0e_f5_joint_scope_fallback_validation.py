"""HZ-0F F5: fallback isolation at the full 3-layer joint scope.

See `docs/restart/hz0e_f5_joint_scope_fallback_validation_results.md`
for the full writeup. Locks in the real, honest, MIXED finding: unlike
F4's single-layer result, `broad_only` does NOT reliably reduce the
general/OOD gap at joint scope (worse than `current` in 2 of 3 seeds),
while `frozen`'s consistently-worse pattern and domain-win preservation
DO generalize. This test intentionally does NOT assert "broad_only
wins" -- that would misrepresent the real result.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from reference.hz0e_f4_fallback_isolation import (
    evaluate_joint_arm, train_joint_dense_baseline_with_general_eval, train_joint_moe_with_fallback_policy,
)
from reference.hz0e_moe_contract import MoeConfig
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists() or not Path(GENERAL_DATA_PATH).exists(),
    reason="real HZ-0A checkpoint/corpus not present locally",
)


def test_joint_scope_frozen_stays_worse_but_broad_only_does_not_reliably_beat_current():
    model, _ = load_frozen_model()
    config = MoeConfig(dim=model.dim)

    print("\nHZ-0F F5 joint 3-layer (27/28/30) fallback validation (real checkpoint, full 50/50/50 protocol):")
    current_gaps, frozen_gaps, broad_only_gaps = [], [], []
    for seed in (0, 1, 2):
        dense_per_domain, dense_general = train_joint_dense_baseline_with_general_eval(model, seed=seed)
        gaps = {}
        for policy in ("current", "frozen", "broad_only"):
            trained = train_joint_moe_with_fallback_policy(model, config, fallback_policy=policy, seed=seed)
            ev = evaluate_joint_arm(model, trained, dense_per_domain, dense_general)
            gap = ev.moe_general_loss - ev.dense_general_loss
            gaps[policy] = gap
            assert ev.domain_win_count >= 3, f"domain win count regressed unexpectedly: {ev.domain_win_count}/5"
            print(f"  seed={seed} {policy}: domain_win={ev.domain_win_count}/5  gap={gap:.4f}")

        current_gaps.append(gaps["current"])
        frozen_gaps.append(gaps["frozen"])
        broad_only_gaps.append(gaps["broad_only"])

    # Real finding #1 (generalizes from F4): frozen is consistently worse than current.
    assert all(f > c for f, c in zip(frozen_gaps, current_gaps)), f"frozen should stay worse than current at joint scope: frozen={frozen_gaps} current={current_gaps}"

    # Real finding #2 (does NOT generalize from F4): broad_only is NOT
    # consistently better than current at joint scope -- this asserts the
    # HONEST mixed result (at least one seed where it fails to improve),
    # not the single-layer win.
    assert any(bo >= cu for bo, cu in zip(broad_only_gaps, current_gaps)), (
        f"expected the real joint-scope result (broad_only does not reliably beat current) to hold: "
        f"broad_only={broad_only_gaps} current={current_gaps} -- if this now passes with broad_only always "
        f"better, the joint-scope finding needs re-examination, not silent re-assertion"
    )
