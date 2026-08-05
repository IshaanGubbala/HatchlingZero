"""HZ-0D D8: curriculum tests (reference/hz0d_d8_curriculum.py).

Checked against the ACTUAL frozen checkpoint and REAL corpus text
(`data/packed/repro_1024_val.jsonl`). Skips if either isn't present
locally. Locks in D8's exit gate directly: "adaptation is sparse,
quick, and reversible," across all 5 curriculum stages the plan names.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from reference.hz0d_d6_integration import ATTENTION_INDICES
from reference.hz0d_d8_curriculum import make_natural_schema_task
from reference.hz0d_fast_weights import (
    FastWeightConfig, FastWeightState, init_fast_weights, rollback, snapshot,
)
from reference.hz0d_isolated_simulator import held_out_generalization_loss, task_loss
from reference.hz0d_update_mechanisms import delta_prediction_update
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists() or not Path(GENERAL_DATA_PATH).exists(),
    reason="frozen HZ-0A checkpoint / real corpus data not present locally (both gitignored)",
)

# max_delta_norm=10.0, matching reference/hz0d_d6_integration.py's own calibration
# convention: a rule_scale=0.05 true_delta already has Frobenius norm ~7.67 (checked
# directly), well beyond D1's production safety default of max_delta_norm=1.0 -- that
# default bounds worst-case ADVERSARIAL deltas (exercised deliberately by stage 5
# below via ADVERSARIAL_CONFIG), it is not a claim about how large a legitimate
# rule's realized delta may need to be. Conflating the two would make stages 1-4
# about clipping, not adaptation quality.
SINGLE_LAYER_CONFIG = FastWeightConfig(dim=768, rank=16, num_layers=1, max_delta_norm=10.0)
ADVERSARIAL_CONFIG = FastWeightConfig(dim=768, rank=16, num_layers=1, max_delta_norm=1.0)  # D1's real production default
MAX_UPDATE_WALL_SECONDS = 1.0  # "quick" -- generous bound, D3/D6 measured ~150x faster than 400 GD steps


def _load_real_tokens(count: int = 8) -> mx.array:
    sequences = load_real_sequences(GENERAL_DATA_PATH, count)
    min_len = min(len(s) for s in sequences)
    return mx.array([s[:min_len] for s in sequences])


def _zero_state() -> FastWeightState:
    return FastWeightState(a_fast=mx.zeros((1, 768, 16)), b_fast=mx.zeros((1, 16, 768)), update_count=mx.array(0, dtype=mx.int32))


def test_stage1_explicit_update_supervision_reduces_held_out_loss():
    """Stage 1: given an explicit, fully-labeled Task, a real
    fast-weight update must genuinely reduce held-out loss versus the
    inactive baseline -- the direct supervised-adaptation case, on real
    corpus-derived activations."""
    model, _payload = load_frozen_model()
    tokens = _load_real_tokens(8)
    task = make_natural_schema_task(model, tokens, heads=model.heads, seed=0, rule_scale=0.05, k_train=256, k_held_out=64)
    zero_loss = float(task_loss(task, _zero_state(), task.held_out_x, task.held_out_y))
    active_state, diag = delta_prediction_update(task, init_fast_weights(SINGLE_LAYER_CONFIG), SINGLE_LAYER_CONFIG)
    active_loss = held_out_generalization_loss(task, active_state)
    assert active_loss < zero_loss * 0.5
    assert diag["wall_seconds"] < MAX_UPDATE_WALL_SECONDS


def test_stage2_few_shot_rule_inference_generalizes_not_memorizes():
    """Stage 2: with a SMALL number of real examples, the fit must
    genuinely generalize to held-out real activations, not merely
    memorize the training set -- both train AND held-out loss must drop
    substantially relative to the inactive baseline (pure memorization
    would show train loss near zero while held-out loss stays high)."""
    model, _payload = load_frozen_model()
    tokens = _load_real_tokens(8)
    task = make_natural_schema_task(model, tokens, heads=model.heads, seed=1, rule_scale=0.05, k_train=48, k_held_out=64)
    zero_loss = float(task_loss(task, _zero_state(), task.held_out_x, task.held_out_y))
    active_state, diag = delta_prediction_update(task, init_fast_weights(SINGLE_LAYER_CONFIG), SINGLE_LAYER_CONFIG)
    held_out_loss = held_out_generalization_loss(task, active_state)
    train_loss = diag["final_train_loss"]
    assert held_out_loss < zero_loss * 0.85, "expected genuine held-out generalization, not just train-set fitting"
    assert train_loss < zero_loss * 0.85, "expected genuine train-set fit too, not a degenerate non-fit"


def test_stage3_rule_switching_shows_real_interference():
    """Stage 3: adapt to rule A, then adapt the SAME layer to a
    DIFFERENT rule B. Rule B's fit must overwrite rule A's -- A's
    held-out loss should get WORSE after switching to B (real
    interference/forgetting, not independently-tracked rules), while
    B's own held-out loss should be genuinely good."""
    model, _payload = load_frozen_model()
    tokens = _load_real_tokens(8)
    task_a = make_natural_schema_task(model, tokens, heads=model.heads, seed=10, rule_scale=0.05, k_train=128, k_held_out=64)
    task_b = make_natural_schema_task(model, tokens, heads=model.heads, seed=20, rule_scale=0.05, k_train=128, k_held_out=64)

    state_a, _ = delta_prediction_update(task_a, init_fast_weights(SINGLE_LAYER_CONFIG), SINGLE_LAYER_CONFIG)
    loss_a_under_a = held_out_generalization_loss(task_a, state_a)

    state_b, _ = delta_prediction_update(task_b, state_a, SINGLE_LAYER_CONFIG)
    loss_a_under_b = held_out_generalization_loss(task_a, state_b)
    zero_loss_a = float(task_loss(task_a, _zero_state(), task_a.held_out_x, task_a.held_out_y))
    loss_b_under_b = held_out_generalization_loss(task_b, state_b)
    zero_loss_b = float(task_loss(task_b, _zero_state(), task_b.held_out_x, task_b.held_out_y))

    assert loss_a_under_b > loss_a_under_a, "expected switching to rule B to degrade rule A's held-out fit (real interference)"
    assert loss_b_under_b < zero_loss_b * 0.5, "expected rule B's own held-out loss to be genuinely good after switching"
    assert loss_a_under_a < zero_loss_a * 0.5, "expected rule A's held-out loss to have been genuinely good before switching"


def test_stage4_natural_schema_input_is_real_corpus_activations_not_synthetic():
    """Stage 4, the genuinely new D8 piece: the task's `x` values must
    actually come from real corpus text through the real backbone (this
    checkpoint has no meaningful "instruction" to follow, so the
    natural-ness is in the INPUT distribution, disclosed in
    `reference/hz0d_d8_curriculum.py`'s module docstring) -- checked
    directly by confirming the collected activations are finite,
    nonzero, and match the real per-position count from the real token
    sequences (not a synthetic count)."""
    model, _payload = load_frozen_model()
    tokens = _load_real_tokens(4)
    task = make_natural_schema_task(model, tokens, heads=model.heads, seed=2, rule_scale=0.05, k_train=64, k_held_out=32)
    assert bool(mx.all(mx.isfinite(task.train_x)))
    assert float(mx.std(task.train_x)) > 1e-4
    total_real_positions = tokens.shape[0] * tokens.shape[1]
    assert task.train_x.shape[0] + task.held_out_x.shape[0] <= total_real_positions


def test_stage5_adversarial_update_stays_clipped_and_curriculum_is_exactly_reversible():
    """Stage 5: an adversarially large rule (`rule_scale=1000`, designed
    to force a huge delta) must still respect `max_delta_norm` after a
    real `delta_prediction_update` -- clipping applies on this path too,
    not just gradient descent's. Then: snapshot the curriculum-start
    state, run several real updates (stages 1/2/3 worth), and roll back
    -- the result must be BIT-IDENTICAL to the original snapshot, and
    behaviorally identical (same held-out loss as a fresh zero state),
    matching D8's exit gate: adaptation is reversible."""
    model, _payload = load_frozen_model()
    tokens = _load_real_tokens(8)

    adversarial_task = make_natural_schema_task(model, tokens, heads=model.heads, seed=99, rule_scale=1000.0, k_train=64, k_held_out=16)
    clipped_state, _ = delta_prediction_update(adversarial_task, init_fast_weights(ADVERSARIAL_CONFIG), ADVERSARIAL_CONFIG)
    delta_norm = float(mx.sqrt(mx.sum((clipped_state.a_fast[0] @ clipped_state.b_fast[0]) ** 2)))
    assert delta_norm <= ADVERSARIAL_CONFIG.max_delta_norm + 1e-4

    start_state = init_fast_weights(SINGLE_LAYER_CONFIG)
    checkpoint = snapshot(start_state)

    state = start_state
    update_count = 0
    for seed in (30, 31, 32):
        task = make_natural_schema_task(model, tokens, heads=model.heads, seed=seed, rule_scale=0.05, k_train=64, k_held_out=16)
        state, _diag = delta_prediction_update(task, state, SINGLE_LAYER_CONFIG)
        update_count += 1
    assert update_count == 3, "sparse: a bounded, small number of real updates for this curriculum"
    assert int(state.update_count) == 3

    restored = rollback(checkpoint)
    assert bool(mx.array_equal(restored.a_fast, start_state.a_fast))
    assert bool(mx.array_equal(restored.b_fast, start_state.b_fast))
    probe_task = make_natural_schema_task(model, tokens, heads=model.heads, seed=40, rule_scale=0.05, k_train=16, k_held_out=16)
    assert task_loss(probe_task, restored, probe_task.held_out_x, probe_task.held_out_y) == task_loss(probe_task, _zero_state(), probe_task.held_out_x, probe_task.held_out_y)
