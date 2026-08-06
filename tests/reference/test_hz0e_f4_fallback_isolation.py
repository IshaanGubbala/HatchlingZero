"""HZ-0F F4: three-arm fallback isolation experiment
(`reference/hz0e_f4_fallback_isolation.py`).

See `docs/restart/hz0e_f4_fallback_isolation_results.md` for the full
writeup. Locks in the real, multi-seed, reproducible DIRECTION of the
headline finding: training the shared fallback ONLY on separate
general-prose replay batches (not curriculum-domain overflow gradients)
flips the general/OOD quality gap from a consistent MoE deficit to a
consistent MoE advantage, while a plain frozen fallback does neither
(and makes the OOD gap worse) -- ruling out "incidental curriculum
training corrupts the fallback" in favor of "the fallback needs its own
dedicated general training."
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from reference.hz0e_e2_router_simulator import DOMAIN_DATA_PATHS
from reference.hz0e_e8_curriculum import LAYER, load_domain_batches, run_warm_dense_baseline
from reference.hz0e_f2_gate_overflow_audit import fallback_vs_dense_audit
from reference.hz0e_f4_fallback_isolation import evaluate_arm, train_moe_with_fallback_policy
from reference.hz0e_moe_contract import MoeConfig
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists() or not Path(GENERAL_DATA_PATH).exists(),
    reason="real HZ-0A checkpoint/corpus not present locally",
)


def test_frozen_fallback_is_bit_identical_to_warm_start():
    """Correctness precondition for the whole experiment: `frozen`
    must genuinely never update the fallback, not merely train it
    slowly. Checked directly, not assumed."""
    from reference.hz0e_e2_router_simulator import DOMAIN_DATA_PATHS as _paths
    from reference.hz0e_e3_routing_objectives import supervised_warm_start
    from reference.hz0e_e6_integration import init_e6_layers
    from reference.hz0e_e8_curriculum import DOMAIN_TO_EXPERT, TRAIN_DOMAIN_DATA_PATHS

    model, _ = load_frozen_model()
    config = MoeConfig(dim=model.dim)
    train_domains = load_domain_batches(TRAIN_DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=0)
    e6_layers = init_e6_layers(model, seed=0)
    warm = supervised_warm_start(model, train_domains, DOMAIN_TO_EXPERT, config, layer_index=LAYER, steps=40, learning_rate=1e-3, start_params=e6_layers[LAYER], cache_backbone=True)

    frozen_trained = train_moe_with_fallback_policy(model, config, fallback_policy="frozen", seed=0)

    assert bool(mx.array_equal(warm.fallback_gate_w, frozen_trained.fallback_gate_w))
    assert bool(mx.array_equal(warm.fallback_down_b, frozen_trained.fallback_down_b))
    assert not bool(mx.array_equal(warm.router_w, frozen_trained.router_w)), "router should still train"
    assert not bool(mx.array_equal(warm.expert_gate_w, frozen_trained.expert_gate_w)), "experts should still train"


def test_broad_only_fallback_flips_the_ood_gap_and_frozen_does_not():
    model, _ = load_frozen_model()
    config = MoeConfig(dim=model.dim)
    held_out_domains = load_domain_batches(DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=1)
    in_distribution_batch = held_out_domains["prose"]
    ood_batch = mx.concatenate([mx.array([s[:64]]) for s in load_real_sequences(GENERAL_DATA_PATH, 8)], axis=0)

    print("\nHZ-0F F4 three-arm fallback isolation (real checkpoint, full 50/50/50 protocol):")
    broad_only_ood_gaps = []
    current_ood_gaps = []
    frozen_ood_gaps = []
    broad_only_differentials = []
    current_differentials = []

    for seed in (0, 1, 2):
        dense_trained, _before, _after = run_warm_dense_baseline(model, seed=seed)
        results = {}
        for policy in ("current", "frozen", "broad_only"):
            moe_trained = train_moe_with_fallback_policy(model, config, fallback_policy=policy, seed=seed)
            ev = evaluate_arm(model, moe_trained, dense_trained, config)
            ood_gap = ev.moe_general_loss - ev.dense_general_loss
            in_fb = fallback_vs_dense_audit(model, moe_trained, dense_trained, config, LAYER, in_distribution_batch)
            ood_fb = fallback_vs_dense_audit(model, moe_trained, dense_trained, config, LAYER, ood_batch)
            differential = ood_fb.fallback_minus_dense_gap - in_fb.fallback_minus_dense_gap
            results[policy] = (ood_gap, differential, ev.domain_win_count)
            print(f"  seed={seed} {policy}: domain_win={ev.domain_win_count}/5  ood_gap={ood_gap:.4f}  fb_differential={differential:.4f}")

        current_ood_gaps.append(results["current"][0])
        frozen_ood_gaps.append(results["frozen"][0])
        broad_only_ood_gaps.append(results["broad_only"][0])
        current_differentials.append(results["current"][1])
        broad_only_differentials.append(results["broad_only"][1])

        assert results["broad_only"][2] >= 3, "broad_only should preserve most of the in-domain win count"

    # Real, reproducible findings, all 3 seeds:
    assert all(gap < 0.0 for gap in broad_only_ood_gaps), f"broad_only should flip the OOD gap net-positive for MoE in every seed: {broad_only_ood_gaps}"
    assert all(f > c for f, c in zip(frozen_ood_gaps, current_ood_gaps)), f"frozen should NOT help the OOD gap (should be worse than current): frozen={frozen_ood_gaps} current={current_ood_gaps}"
    assert all(bo < cu for bo, cu in zip(broad_only_differentials, current_differentials)), f"broad_only should reduce the fallback differential vs current: broad_only={broad_only_differentials} current={current_differentials}"
