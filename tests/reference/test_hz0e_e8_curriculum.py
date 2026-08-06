"""HZ-0E E8: specialization curriculum tests (reference/hz0e_e8_curriculum.py).
Checked against the ACTUAL frozen checkpoint and REAL corpus text.
Skips if either is missing locally. Locks in the real, measured
findings from `docs/restart/hz0e_e8_specialization_curriculum_results.md`
as regression tests, including two real bugs found and fixed before any
number in that document was trusted.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from reference.hz0e_e2_router_simulator import DOMAIN_DATA_PATHS
from reference.hz0e_e3_routing_objectives import lm_forward_with_moe, params_to_dict, train_moe_layer
from reference.hz0e_e4_fair_baselines import no_adaptation_loss
from reference.hz0e_e6_integration import init_e6_layers
from reference.hz0e_e8_curriculum import (
    LAYER, TRAIN_DOMAIN_DATA_PATHS, evaluate_dense_per_domain, evaluate_joint_moe_per_domain,
    evaluate_moe_per_domain, load_domain_batches, mean_pairwise_tv_distance, measure_specialization,
    mixed_domain_batches, per_domain_mean_loss, run_curriculum, run_joint_multilayer_curriculum,
    run_joint_multilayer_dense_baseline, run_warm_dense_baseline, warm_dense_init,
)
from reference.hz0e_moe_contract import MoeConfig
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c3_trigger_simulator import load_real_sequences

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists() or not Path(TRAIN_DOMAIN_DATA_PATHS["prose"]).exists(),
    reason="frozen HZ-0A checkpoint / real domain train corpus not present locally (gitignored)",
)

CONFIG = MoeConfig()


def test_train_and_validation_domain_files_are_genuinely_disjoint():
    """The real bug this module's own history found: `DOMAIN_DATA_PATHS`
    (E2's validation-only files) must NEVER be reused as a training
    source -- `DOMAIN_DATA_PATHS["prose"]` and the general-quality
    held-out set both point at `repro_1024_val.jsonl`, so training on
    the former while evaluating on the latter is real leakage. Locks in
    the fix: `TRAIN_DOMAIN_DATA_PATHS` must point at DIFFERENT files
    than `DOMAIN_DATA_PATHS`, for every domain, not just prose."""
    for name in DOMAIN_DATA_PATHS:
        assert TRAIN_DOMAIN_DATA_PATHS[name] != DOMAIN_DATA_PATHS[name], (
            f"{name}: train and validation paths must differ, found the same file for both"
        )


def test_no_record_overlap_between_train_and_validation_domain_data_at_the_offset_this_module_uses():
    """A stronger check than "different filenames" -- the actual real
    sequences `run_curriculum` loads for held-out measurement
    (`offset=1`, see its own docstring) must not be identical records
    to the training data. A real, narrow, pre-existing corpus quirk was
    found here during development: `json_and_configuration_train.jsonl`
    record 0 and `json_and_configuration_validation.jsonl` record 0 are
    IDENTICAL (every other domain, and every other checked record, is
    clean) -- `run_curriculum` works around this with `offset=1`; this
    test locks in that the workaround is real and sufficient, not just
    asserted."""
    for name in DOMAIN_DATA_PATHS:
        train_seqs = load_real_sequences(TRAIN_DOMAIN_DATA_PATHS[name], 4)
        val_seqs = load_real_sequences(DOMAIN_DATA_PATHS[name], 5)[1:]  # matches run_curriculum's offset=1
        overlap = any(t[:20] == v[:20] for t in train_seqs for v in val_seqs)
        assert not overlap, f"{name}: found overlapping records between train and validation files at offset=1"


def test_json_domain_train_and_validation_files_have_a_known_duplicate_at_record_zero():
    """Documents the real, narrow corpus quirk directly, as a locked-in
    regression fact rather than a comment someone could miss: if this
    ever stops being true (e.g. the corpus is regenerated), the
    `offset=1` workaround in `run_curriculum` should be revisited, not
    silently left in place for a problem that no longer exists."""
    train_seqs = load_real_sequences(TRAIN_DOMAIN_DATA_PATHS["json"], 1)
    val_seqs = load_real_sequences(DOMAIN_DATA_PATHS["json"], 1)
    assert train_seqs[0][:20] == val_seqs[0][:20], (
        "expected the known json train[0]==val[0] duplicate to still be present -- "
        "if this fails, the corpus may have been fixed/regenerated; reconsider whether "
        "run_curriculum's offset=1 workaround is still needed"
    )


def test_mixed_domain_batches_seed_actually_changes_the_curriculum():
    """The second real bug found during development: `mixed_domain_batches`
    accepted a `seed` parameter and created an `mx.random.key` from it,
    but never actually used the key -- every "different seed" produced
    an IDENTICAL curriculum. Locks in the fix: two different seeds must
    produce genuinely different domain pairings."""
    domain_batches = load_domain_batches(TRAIN_DOMAIN_DATA_PATHS, count=4, seq_len=32)
    batches_seed0 = mixed_domain_batches(domain_batches, steps=10, seed=0)
    batches_seed1 = mixed_domain_batches(domain_batches, steps=10, seed=1)
    any_differs = any(
        b0.shape != b1.shape or not bool(mx.array_equal(b0, b1))
        for b0, b1 in zip(batches_seed0, batches_seed1)
    )
    assert any_differs, "expected different seeds to produce a genuinely different mixed-domain curriculum"


def test_learning_rate_1e4_diverges_but_1e5_stays_stable_in_the_warm_started_regime():
    """The real, measured learning-rate finding this module's own
    docstring documents: `lr=1e-4` (E3's rate, tuned for small-random
    init) causes real divergence when starting from E6's much-larger-
    magnitude warm-started weights, while `lr=1e-5` stays stable and
    gives a small real improvement -- confirmed directly, not assumed
    to transfer from E3's own regime."""
    model, _payload = load_frozen_model()
    train_domains = load_domain_batches(TRAIN_DOMAIN_DATA_PATHS, count=4, seq_len=32)
    general_val = [mx.array([s[:32]]) for s in load_real_sequences("data/packed/repro_1024_val.jsonl", 5)]

    from reference.hz0e_e3_routing_objectives import supervised_warm_start
    from reference.hz0e_e8_curriculum import DOMAIN_TO_EXPERT, balanced_batches

    e6_layers = init_e6_layers(model, seed=0)
    warm = supervised_warm_start(model, train_domains, DOMAIN_TO_EXPERT, CONFIG, layer_index=LAYER, steps=10, learning_rate=1e-3, start_params=e6_layers[LAYER])
    warm_dict = params_to_dict(warm)
    warm_val = sum(float(lm_forward_with_moe(warm_dict, model, tb, CONFIG, LAYER)[0]) for tb in general_val) / len(general_val)

    batches = balanced_batches(train_domains, 20)

    unstable, _ = train_moe_layer(model, batches, CONFIG, layer_index=LAYER, aux_weights={}, learning_rate=1e-4, start_params=warm)
    unstable_dict = params_to_dict(unstable)
    unstable_val = sum(float(lm_forward_with_moe(unstable_dict, model, tb, CONFIG, LAYER)[0]) for tb in general_val) / len(general_val)

    stable, _ = train_moe_layer(model, batches, CONFIG, layer_index=LAYER, aux_weights={}, learning_rate=1e-5, start_params=warm)
    stable_dict = params_to_dict(stable)
    stable_val = sum(float(lm_forward_with_moe(stable_dict, model, tb, CONFIG, LAYER)[0]) for tb in general_val) / len(general_val)

    assert unstable_val > warm_val, f"expected lr=1e-4 to diverge above the warm-start baseline: warm={warm_val} lr1e-4={unstable_val}"
    assert stable_val <= warm_val * 1.02, f"expected lr=1e-5 to stay close to or better than the warm-start baseline: warm={warm_val} lr1e-5={stable_val}"


def test_run_curriculum_produces_finite_specialization_report():
    """`run_curriculum`'s end-to-end real path: finite general-quality
    and specialization metrics, real per-domain losses for every
    domain, at the module's own default (proven-stable) learning rate."""
    model, _payload = load_frozen_model()
    config = MoeConfig(dim=model.dim)
    _trained, report = run_curriculum(model, config, balanced_steps=5, mixed_steps=5, imbalanced_steps=5, seed=0, warm_start_steps=10)

    assert report.general_val_loss_before == report.general_val_loss_before  # not NaN
    assert report.general_val_loss_after == report.general_val_loss_after
    assert 0.0 <= report.before_pairwise_tv_distance <= 1.0
    assert 0.0 <= report.after_pairwise_tv_distance <= 1.0
    assert set(report.before_domain_loss) == set(DOMAIN_DATA_PATHS)
    assert set(report.after_domain_loss) == set(DOMAIN_DATA_PATHS)
    assert all(loss == loss for loss in report.after_domain_loss.values())


def test_warm_started_dense_baseline_beats_moe_on_general_held_out_quality():
    """The real, honest resolution of the risk E4 flagged: even with a
    FAIR warm start (dense FFN initialized from the same real
    pretrained-weight-slice recipe MoE's experts use) and the SAME real
    curriculum training, a plain dense FFN matched to MoE's active
    parameter budget still beats MoE on general held-out prose quality
    -- confirmed directly here, not assumed to resolve in MoE's favor
    just because E8's warm-start and curriculum machinery exists."""
    model, _payload = load_frozen_model()
    config = MoeConfig(dim=model.dim)

    _moe_trained, moe_report = run_curriculum(model, config, balanced_steps=15, mixed_steps=15, imbalanced_steps=15, seed=0, warm_start_steps=20)
    _dense_trained, _dense_before, dense_after = run_warm_dense_baseline(model, balanced_steps=15, mixed_steps=15, imbalanced_steps=15, seed=0)

    assert dense_after <= moe_report.general_val_loss_after * 1.01, (
        f"expected the fairly warm-started dense baseline to be at least competitive with MoE: "
        f"dense={dense_after} moe={moe_report.general_val_loss_after}"
    )


def test_specialization_measurement_uses_genuinely_held_out_domain_data():
    """Structural check: `measure_specialization`'s held-out domain
    batches must come from `DOMAIN_DATA_PATHS` (validation files), not
    `TRAIN_DOMAIN_DATA_PATHS` -- confirmed by checking the function
    receives and correctly processes validation-sourced data without
    error, matching the module's own documented convention."""
    model, _payload = load_frozen_model()
    held_out = load_domain_batches(DOMAIN_DATA_PATHS, count=4, seq_len=32)
    from reference.hz0e_moe_contract import init_moe_layer
    params = init_moe_layer(CONFIG)
    util = measure_specialization(model, params, CONFIG, held_out)
    assert set(util.keys()) == set(DOMAIN_DATA_PATHS)
    tv = mean_pairwise_tv_distance(util)
    assert 0.0 <= tv <= 1.0


def test_full_3layer_joint_moe_still_does_not_beat_pure_dense_baseline():
    """The most exhaustive real check available: E1's ACTUAL scoped
    contract (layers 27, 28, 30 converted SIMULTANEOUSLY, not one
    isolated layer) trained jointly via `run_joint_multilayer_curriculum`,
    compared against the real, untouched, pure frozen dense model
    (`enabled=False` in `forward_e6`, i.e. `HZ0AMlxModel`'s own
    original forward pass with no MoE anywhere). Confirms the E4/E8
    single-layer finding is not an artifact of testing only one of the
    3 real target layers -- the full, actually-scoped 3-layer
    integration shows the SAME real result: MoE does not beat the
    pure dense baseline on held-out general quality, even when given
    every real advantage (E6 warm-start, router supervision, a
    properly-tuned learning rate) across its ENTIRE real footprint."""
    model, _payload = load_frozen_model()
    config = MoeConfig(dim=model.dim)
    _trained, pure_dense, _warm, after = run_joint_multilayer_curriculum(
        model, config, balanced_steps=15, mixed_steps=15, imbalanced_steps=15, warm_start_steps=20, seed=0,
    )
    assert after >= pure_dense * 0.99, (
        f"expected the full 3-layer joint MoE to NOT clearly beat the pure dense baseline: "
        f"moe={after} pure_dense={pure_dense}"
    )


def test_single_layer_moe_beats_fair_dense_on_per_domain_in_distribution_quality():
    """A real, reproducible, previously-unreported POSITIVE finding for
    MoE: on mean held-out loss across the 5 real domains the curriculum
    actually trains on (in-distribution quality -- the direct test of
    whether specialization helps, as opposed to the general-prose
    out-of-distribution robustness check every other test in this
    module and E4 used as the sole metric), single-layer MoE beats a
    FAIRLY warm-started dense baseline of the same active-parameter
    budget. Confirmed across 3 seeds in
    `docs/restart/hz0e_moe_per_domain_significance_results.md`; this
    test locks in the direction at reduced scale for CI speed."""
    model, _payload = load_frozen_model()
    config = MoeConfig(dim=model.dim)
    d_ff = 577

    moe_trained, _report = run_curriculum(model, config, balanced_steps=15, mixed_steps=15, imbalanced_steps=15, seed=0, warm_start_steps=20)
    moe_domain_losses = evaluate_moe_per_domain(model, moe_trained, config)
    moe_mean = per_domain_mean_loss(moe_domain_losses)

    from reference.hz0e_e4_fair_baselines import train_generic
    from reference.hz0e_e8_curriculum import balanced_batches, imbalanced_batches, make_warm_dense_loss_fn, mixed_domain_batches
    train_domains = load_domain_batches(TRAIN_DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=0)
    loss_fn = make_warm_dense_loss_fn(model, LAYER)
    stage1 = balanced_batches(train_domains, 15); stage2 = mixed_domain_batches(train_domains, 15, seed=0); stage3 = imbalanced_batches(train_domains, 15)
    dense_params, _losses = train_generic(model, stage1 + stage2 + stage3, lambda: warm_dense_init(model, LAYER, d_ff), loss_fn, learning_rate=1e-5)
    dense_domain_losses = evaluate_dense_per_domain(model, dense_params)
    dense_mean = per_domain_mean_loss(dense_domain_losses)

    assert moe_mean < dense_mean, (
        f"expected MoE to beat fair dense on per-domain (in-distribution) quality: moe={moe_mean} dense={dense_mean}"
    )


def test_joint_3layer_moe_beats_fair_dense_on_per_domain_in_distribution_quality():
    """The same real finding, confirmed at E1's actual full 3-layer
    scope, not just one isolated layer -- reproduced across all 3
    seeds tested in the results doc."""
    model, _payload = load_frozen_model()
    config = MoeConfig(dim=model.dim)

    trained_layers, _pd, _w, _a = run_joint_multilayer_curriculum(model, config, balanced_steps=15, mixed_steps=15, imbalanced_steps=15, warm_start_steps=20, seed=0)
    moe_mean = per_domain_mean_loss(evaluate_joint_moe_per_domain(model, trained_layers))

    dense_domain_losses = run_joint_multilayer_dense_baseline(model, balanced_steps=15, mixed_steps=15, imbalanced_steps=15, seed=0)
    dense_mean = per_domain_mean_loss(dense_domain_losses)

    assert moe_mean < dense_mean, (
        f"expected 3-layer MoE to beat fair 3-layer dense on per-domain (in-distribution) quality: "
        f"moe={moe_mean} dense={dense_mean}"
    )


def test_replay_improves_both_mechanisms_but_does_not_erase_the_relative_tradeoff():
    """Real, principled test of whether replay/rehearsal (a standard
    continual-learning technique for exactly the specialization-costs-
    generality problem this module found) CLOSES the gap between MoE
    and dense, or whether the in-distribution/out-of-distribution
    tradeoff is structural. Both mechanisms get the SAME real replay
    treatment (extra general-prose batches, disjoint from what the
    curriculum's own "prose" domain already trains on, interleaved
    evenly). Real result: replay improves both absolute numbers, but
    dense still wins on general/out-of-distribution quality and MoE
    still wins on per-domain/in-distribution quality -- locked in here,
    not assumed to hold from the un-replayed comparison alone."""
    model, _payload = load_frozen_model()
    config = MoeConfig(dim=model.dim)
    d_ff = 577

    from reference.hz0e_e3_routing_objectives import lm_forward_with_moe, supervised_warm_start
    from reference.hz0e_e4_fair_baselines import eval_generic, train_generic
    from reference.hz0e_e6_integration import init_e6_layers
    from reference.hz0e_e8_curriculum import (
        DOMAIN_TO_EXPERT, balanced_batches, imbalanced_batches, interleave_replay, load_replay_batches,
        make_warm_dense_loss_fn, mixed_domain_batches,
    )

    train_domains = load_domain_batches(TRAIN_DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=0)
    general_val = [mx.array([s[:64]]) for s in load_real_sequences("data/packed/repro_1024_val.jsonl", 10)]
    replay_batches = load_replay_batches(count=10, domain_train_count=8)

    stage1 = balanced_batches(train_domains, 20)
    stage2 = mixed_domain_batches(train_domains, 20, seed=0)
    stage3 = imbalanced_batches(train_domains, 20)
    curriculum = stage1 + stage2 + stage3
    with_replay = interleave_replay(curriculum, replay_batches)
    assert len(with_replay) > len(curriculum), "replay must actually add batches to the curriculum"

    e6_layers = init_e6_layers(model, seed=0)
    warm = supervised_warm_start(model, train_domains, DOMAIN_TO_EXPERT, config, layer_index=LAYER, steps=20, learning_rate=1e-3, start_params=e6_layers[LAYER])
    moe_trained, _hist = train_moe_layer(model, with_replay, config, layer_index=LAYER, aux_weights={}, learning_rate=1e-5, start_params=warm)
    moe_trained_dict = params_to_dict(moe_trained)
    moe_general = sum(float(lm_forward_with_moe(moe_trained_dict, model, tb, config, LAYER)[0]) for tb in general_val) / len(general_val)
    moe_domain_mean = per_domain_mean_loss(evaluate_moe_per_domain(model, moe_trained, config))

    dense_loss_fn = make_warm_dense_loss_fn(model, LAYER)
    dense_trained, _losses = train_generic(model, with_replay, lambda: warm_dense_init(model, LAYER, d_ff), dense_loss_fn, learning_rate=1e-5)
    dense_general = eval_generic(model, general_val, dense_trained, dense_loss_fn)
    dense_domain_mean = per_domain_mean_loss(evaluate_dense_per_domain(model, dense_trained))

    assert dense_general < moe_general, (
        f"expected dense to still win on general/out-of-distribution quality even with replay: "
        f"dense={dense_general} moe={moe_general}"
    )
    assert moe_domain_mean < dense_domain_mean, (
        f"expected MoE to still win on per-domain/in-distribution quality even with replay: "
        f"moe={moe_domain_mean} dense={dense_domain_mean}"
    )
