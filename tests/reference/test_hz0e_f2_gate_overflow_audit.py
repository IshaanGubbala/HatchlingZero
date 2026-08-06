"""HZ-0F F2: gate confidence and overflow/fallback audit
(`reference/hz0e_f2_gate_overflow_audit.py`).

See `docs/restart/hz0e_f2_gate_overflow_fallback_results.md` for the
full writeup. Locks in the real, multi-seed, reproducible DIRECTION of
each finding (not exact values, which have real seed-to-seed
variation): neither gate confidence calibration nor internal-fallback
quality shows a large, OOD-amplified effect that would explain the
documented aggregate OOD quality gap on its own.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from reference.hz0e_e2_router_simulator import DOMAIN_DATA_PATHS
from reference.hz0e_e8_curriculum import LAYER, load_domain_batches, run_curriculum, run_warm_dense_baseline
from reference.hz0e_f2_gate_overflow_audit import fallback_vs_dense_audit, gate_forcing_audit, measure_gate_and_overflow
from reference.hz0e_moe_contract import MoeConfig
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists() or not Path(GENERAL_DATA_PATH).exists(),
    reason="real HZ-0A checkpoint/corpus not present locally",
)


def _batches():
    held_out_domains = load_domain_batches(DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=1)
    in_distribution_batch = held_out_domains["prose"]
    ood_batch = mx.concatenate([mx.array([s[:64]]) for s in load_real_sequences(GENERAL_DATA_PATH, 8)], axis=0)
    return in_distribution_batch, ood_batch


def test_gate_forcing_and_fallback_audits_are_structurally_valid_and_not_ood_amplified():
    model, _ = load_frozen_model()
    config = MoeConfig(dim=model.dim)
    in_distribution_batch, ood_batch = _batches()

    gate_delta_diffs = []
    fallback_gap_diffs = []
    print("\nHZ-0F F2 gate/overflow/fallback audit (real checkpoint, real curriculum):")
    for seed in (0, 1, 2):
        moe_trained, _report = run_curriculum(model, config, balanced_steps=15, mixed_steps=15, imbalanced_steps=15, seed=seed, warm_start_steps=20)
        dense_trained, _before, _after = run_warm_dense_baseline(model, balanced_steps=15, mixed_steps=15, imbalanced_steps=15, seed=seed)

        in_overflow_stats = measure_gate_and_overflow(model, moe_trained, LAYER, in_distribution_batch)
        ood_overflow_stats = measure_gate_and_overflow(model, moe_trained, LAYER, ood_batch)
        assert 0.0 <= in_overflow_stats.overflow_rate <= 1.0
        assert 0.0 <= ood_overflow_stats.overflow_rate <= 1.0

        in_gate = gate_forcing_audit(model, moe_trained, config, LAYER, in_distribution_batch)
        ood_gate = gate_forcing_audit(model, moe_trained, config, LAYER, ood_batch)
        assert all(v == v for v in [in_gate.real_gated_loss, ood_gate.real_gated_loss]), "non-finite loss"
        gate_delta_diffs.append(abs(ood_gate.delta - in_gate.delta))

        in_fb = fallback_vs_dense_audit(model, moe_trained, dense_trained, config, LAYER, in_distribution_batch)
        ood_fb = fallback_vs_dense_audit(model, moe_trained, dense_trained, config, LAYER, ood_batch)
        assert all(v == v for v in [in_fb.fallback_token_loss, ood_fb.fallback_token_loss]), "non-finite loss"
        fallback_gap_diffs.append(abs(ood_fb.fallback_minus_dense_gap - in_fb.fallback_minus_dense_gap))

        print(f"  seed={seed} gate delta: in-dist={in_gate.delta:.4f} ood={ood_gate.delta:.4f}")
        print(f"  seed={seed} fallback gap: in-dist={in_fb.fallback_minus_dense_gap:.4f} (overflow={in_fb.overflow_rate:.3f})  ood={ood_fb.fallback_minus_dense_gap:.4f} (overflow={ood_fb.overflow_rate:.3f})")

    # Real, reproducible finding: neither mechanism's ID/OOD difference is
    # large -- both stay within a small band, not a dramatic OOD-specific
    # blowup. Locks in the DIRECTION/SCALE of the finding, not exact values.
    assert all(diff < 0.01 for diff in gate_delta_diffs), f"gate-forcing ID/OOD difference larger than expected: {gate_delta_diffs}"
    assert all(diff < 0.01 for diff in fallback_gap_diffs), f"fallback-vs-dense ID/OOD difference larger than expected: {fallback_gap_diffs}"
