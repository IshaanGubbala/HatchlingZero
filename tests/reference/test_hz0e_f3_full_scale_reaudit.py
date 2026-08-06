"""HZ-0F F3: full-scale (50/50/50-step) re-audit of F1/F2's findings.

See `docs/restart/hz0e_f3_full_scale_reaudit_results.md` for the full
writeup. Locks in the real, multi-seed, reproducible DIRECTION of two
findings at the SAME training scale the headline E8/E10 aggregate
numbers were measured at (not the faster diagnostic scale F1/F2's own
tests use): (1) the gated-oracle gap remains non-OOD-amplified even
after the oracle gap itself roughly doubles with more training, and
(2) a real, growing, OOD-unfavorable fallback-vs-dense differential
emerges at this scale that was noise-level at the faster diagnostic
scale -- a genuinely new finding, not merely a re-confirmation.

Slower than F1/F2's own tests (full default step counts, not the
`15/15/15` fast-but-real convention) -- real training time is small
(`~25s`/seed) so this is still a reasonable regression test, not
excluded from the default suite.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from reference.hz0e_e2_router_simulator import DOMAIN_DATA_PATHS
from reference.hz0e_e8_curriculum import LAYER, load_domain_batches, run_curriculum, run_warm_dense_baseline
from reference.hz0e_f1_oracle_routing_audit import oracle_routing_audit
from reference.hz0e_f2_gate_overflow_audit import fallback_vs_dense_audit
from reference.hz0e_moe_contract import MoeConfig
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists() or not Path(GENERAL_DATA_PATH).exists(),
    reason="real HZ-0A checkpoint/corpus not present locally",
)


def test_full_scale_gated_oracle_gap_stays_regime_balanced_and_fallback_differential_grows():
    model, _ = load_frozen_model()
    config = MoeConfig(dim=model.dim)

    held_out_domains = load_domain_batches(DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=1)
    in_distribution_batch = held_out_domains["prose"]
    ood_batch = mx.concatenate([mx.array([s[:64]]) for s in load_real_sequences(GENERAL_DATA_PATH, 8)], axis=0)

    gated_gap_diffs = []
    fallback_differentials = []
    print("\nHZ-0F F3 full-scale (50/50/50) re-audit (real checkpoint):")
    for seed in (0, 1, 2):
        moe_trained, _report = run_curriculum(model, config, seed=seed)  # full defaults: 50/50/50, warm_start=40
        dense_trained, _before, _after = run_warm_dense_baseline(model, seed=seed)  # full defaults: 50/50/50

        in_gated = oracle_routing_audit(model, moe_trained, dense_trained, config, LAYER, in_distribution_batch, gate_scaled=True)
        ood_gated = oracle_routing_audit(model, moe_trained, dense_trained, config, LAYER, ood_batch, gate_scaled=True)
        assert in_gated.oracle_gap >= -1e-4 and ood_gated.oracle_gap >= -1e-4
        gated_gap_diffs.append(ood_gated.oracle_gap - in_gated.oracle_gap)

        in_fb = fallback_vs_dense_audit(model, moe_trained, dense_trained, config, LAYER, in_distribution_batch)
        ood_fb = fallback_vs_dense_audit(model, moe_trained, dense_trained, config, LAYER, ood_batch)
        assert all(v == v for v in [in_fb.fallback_token_loss, ood_fb.fallback_token_loss]), "non-finite loss"
        differential = ood_fb.fallback_minus_dense_gap - in_fb.fallback_minus_dense_gap
        fallback_differentials.append(differential)

        print(f"  seed={seed} gated oracle gap: in-dist={in_gated.oracle_gap:.4f} ood={ood_gated.oracle_gap:.4f}")
        print(f"  seed={seed} fallback gap:     in-dist={in_fb.fallback_minus_dense_gap:.4f}  ood={ood_fb.fallback_minus_dense_gap:.4f}  differential={differential:.4f}")

    # Real, reproducible finding #1 (re-confirmed at full scale): the
    # gated-oracle gap stays small between regimes even though the gap
    # itself grows substantially with more training.
    assert all(abs(diff) < 0.02 for diff in gated_gap_diffs), f"gated oracle gap unexpectedly regime-imbalanced at full scale: {gated_gap_diffs}"

    # Real, reproducible, NEW finding at full scale: the fallback-vs-dense
    # differential is consistently positive (OOD worse than in-dist) and
    # meaningfully larger than the ~0.002 noise-level differential F2
    # found at the faster diagnostic scale.
    assert all(diff > 0.0 for diff in fallback_differentials), f"expected the fallback differential to be consistently positive at full scale: {fallback_differentials}"
