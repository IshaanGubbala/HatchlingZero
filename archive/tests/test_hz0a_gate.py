from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hz0.hz0a_gate import evaluate_hz0a_gates


def test_hz0a_gate_reports_current_mixed_status() -> None:
    scorecard = {
        "hybrid": {
            "25": {
                "step": 25,
                "loss": 3.2,
                "tokens_seen_estimate": 12800.0,
                "estimated_train_flops": 6.0 * 100.0 * 12800.0,
                "tokens_per_second": 100.0,
                "associative_recall_accuracy": 0.0,
                "overwrite_retrieval_accuracy": 0.0,
                "protected_memory_accuracy": 0.0,
                "multi_anchor_retrieval_accuracy": 0.0,
                "multi_anchor_anchor_set_accuracy": 0.0,
                "recall_distance_32_accuracy": 0.0,
                "recall_distance_64_accuracy": 0.0,
                "recall_distance_128_accuracy": 0.0,
                "recall_distance_256_accuracy": 0.0,
            },
            "300": {
                "step": 300,
                "loss": 2.4,
                "tokens_seen_estimate": 153600.0,
                "estimated_train_flops": 6.0 * 100.0 * 153600.0,
                "tokens_per_second": 110.0,
                "associative_recall_accuracy": 0.0,
                "overwrite_retrieval_accuracy": 0.0,
                "protected_memory_accuracy": 0.0,
                "multi_anchor_retrieval_accuracy": 0.0,
                "multi_anchor_anchor_set_accuracy": 0.02,
                "recall_distance_32_accuracy": 0.0,
                "recall_distance_64_accuracy": 0.0,
                "recall_distance_128_accuracy": 0.0,
                "recall_distance_256_accuracy": 0.0,
            },
        },
        "baseline": {
            "25": {
                "step": 25,
                "loss": 3.5,
                "tokens_seen_estimate": 12800.0,
                "estimated_train_flops": 6.0 * 100.0 * 12800.0,
                "tokens_per_second": 180.0,
                "associative_recall_accuracy": 0.0,
                "overwrite_retrieval_accuracy": 0.0,
                "protected_memory_accuracy": 0.0,
                "multi_anchor_retrieval_accuracy": 0.0,
                "multi_anchor_anchor_set_accuracy": 0.0,
                "recall_distance_32_accuracy": 0.0,
                "recall_distance_64_accuracy": 0.0,
                "recall_distance_128_accuracy": 0.0,
                "recall_distance_256_accuracy": 0.0,
            },
            "300": {
                "step": 300,
                "loss": 2.9,
                "tokens_seen_estimate": 153600.0,
                "estimated_train_flops": 6.0 * 100.0 * 153600.0,
                "tokens_per_second": 190.0,
                "associative_recall_accuracy": 0.0,
                "overwrite_retrieval_accuracy": 0.0,
                "protected_memory_accuracy": 0.0,
                "multi_anchor_retrieval_accuracy": 0.0,
                "multi_anchor_anchor_set_accuracy": 0.0,
                "recall_distance_32_accuracy": 0.0,
                "recall_distance_64_accuracy": 0.0,
                "recall_distance_128_accuracy": 0.0,
                "recall_distance_256_accuracy": 0.0,
            },
        },
    }
    reference_manifest = {
        "checkpoint_step": 100,
        "effective_tokens_per_optimizer_update": 512,
        "parameter_count": 50,
    }

    result = evaluate_hz0a_gates(
        scorecard=scorecard,
        reference_manifest=reference_manifest,
        reference_loss=2.8,
        required_transformer_step=300,
        min_loss_margin=0.1,
        min_decode_ratio=0.5,
    )

    # Three HZ-0A gates remain. The fourth gate (memory) moved to HZ-0B tracking.
    assert set(result["gates"].keys()) == {
        "beats_36m_at_fair_tokens_per_param",
        "maintains_transformer_advantage_through_horizon",
        "decode_gap_reduced",
    }
    assert result["gates"]["beats_36m_at_fair_tokens_per_param"]["status"] == "pass"
    assert result["gates"]["maintains_transformer_advantage_through_horizon"]["status"] == "pass"
    assert result["gates"]["decode_gap_reduced"]["status"] == "pass"
    # Memory is now HZ-0B tracking, not an HZ-0A gate.
    assert "shows_memory_task_advantage" not in result["gates"]
    assert "memory_metrics" in result["hz0b_tracking"]
    assert result["hz0b_tracking"]["memory_metrics"] is not None
    assert result["ready_to_continue_scaling"] is True


def test_hz0a_gate_requires_fair_tokens_per_parameter_budget() -> None:
    scorecard = {
        "hybrid": {
            "150": {
                "step": 150,
                "loss": 2.6,
                "tokens_seen_estimate": 76800.0,
                "estimated_train_flops": 6.0 * 109_899_648.0 * 76800.0,
                "tokens_per_second": 103.0,
                "associative_recall_accuracy": 0.0,
                "overwrite_retrieval_accuracy": 0.0,
                "protected_memory_accuracy": 0.0,
                "multi_anchor_retrieval_accuracy": 0.0,
                "multi_anchor_anchor_set_accuracy": 0.01,
                "recall_distance_32_accuracy": 0.0,
                "recall_distance_64_accuracy": 0.0,
                "recall_distance_128_accuracy": 0.0,
                "recall_distance_256_accuracy": 0.0,
            }
        },
        "baseline": {
            "150": {
                "step": 150,
                "loss": 2.95,
                "tokens_seen_estimate": 76800.0,
                "estimated_train_flops": 6.0 * 95_937_984.0 * 76800.0,
                "tokens_per_second": 181.0,
                "associative_recall_accuracy": 0.0,
                "overwrite_retrieval_accuracy": 0.0,
                "protected_memory_accuracy": 0.0,
                "multi_anchor_retrieval_accuracy": 0.0,
                "multi_anchor_anchor_set_accuracy": 0.0,
                "recall_distance_32_accuracy": 0.0,
                "recall_distance_64_accuracy": 0.0,
                "recall_distance_128_accuracy": 0.0,
                "recall_distance_256_accuracy": 0.0,
            }
        },
    }
    reference_manifest = {
        "checkpoint_step": 100,
        "effective_tokens_per_optimizer_update": 512,
        "parameter_count": 36_073_344,
    }

    result = evaluate_hz0a_gates(
        scorecard=scorecard,
        reference_manifest=reference_manifest,
        reference_loss=2.8698,
        required_transformer_step=300,
    )

    assert result["gates"]["beats_36m_at_fair_tokens_per_param"]["status"] == "incomplete"
    assert result["gates"]["maintains_transformer_advantage_through_horizon"]["status"] == "incomplete"
    assert result["gates"]["decode_gap_reduced"]["status"] == "pass"
    # Memory is HZ-0B tracking. We surface the best matched advantage we have
    # (here: step 150, multi_anchor_anchor_set_accuracy, hybrid - baseline = 0.01).
    assert result["hz0b_tracking"]["memory_metrics"] is not None
    memory_metrics = result["hz0b_tracking"]["memory_metrics"]
    assert memory_metrics["step"] == 150
    assert memory_metrics["metric"] == "multi_anchor_anchor_set_accuracy"
    assert memory_metrics["hybrid_value"] == 0.01
    assert memory_metrics["baseline_value"] == 0.0
    assert memory_metrics["advantage"] == 0.01


def test_hz0a_gate_returns_none_memory_when_no_common_steps() -> None:
    """memory_metrics must be None when hybrid and baseline share no steps."""
    scorecard = {
        "hybrid": {
            "25": {
                "step": 25,
                "loss": 3.0,
                "tokens_seen_estimate": 12800.0,
                "estimated_train_flops": 6.0 * 100.0 * 12800.0,
                "tokens_per_second": 100.0,
                "associative_recall_accuracy": 0.0,
                "overwrite_retrieval_accuracy": 0.0,
                "protected_memory_accuracy": 0.0,
                "multi_anchor_retrieval_accuracy": 0.0,
                "multi_anchor_anchor_set_accuracy": 0.0,
                "recall_distance_32_accuracy": 0.0,
                "recall_distance_64_accuracy": 0.0,
                "recall_distance_128_accuracy": 0.0,
                "recall_distance_256_accuracy": 0.0,
            },
        },
        "baseline": {
            "50": {
                "step": 50,
                "loss": 3.3,
                "tokens_seen_estimate": 25600.0,
                "estimated_train_flops": 6.0 * 100.0 * 25600.0,
                "tokens_per_second": 180.0,
                "associative_recall_accuracy": 0.0,
                "overwrite_retrieval_accuracy": 0.0,
                "protected_memory_accuracy": 0.0,
                "multi_anchor_retrieval_accuracy": 0.0,
                "multi_anchor_anchor_set_accuracy": 0.0,
                "recall_distance_32_accuracy": 0.0,
                "recall_distance_64_accuracy": 0.0,
                "recall_distance_128_accuracy": 0.0,
                "recall_distance_256_accuracy": 0.0,
            },
        },
    }
    reference_manifest = {
        "checkpoint_step": 100,
        "effective_tokens_per_optimizer_update": 512,
        "parameter_count": 36_073_344,
    }

    result = evaluate_hz0a_gates(
        scorecard=scorecard,
        reference_manifest=reference_manifest,
        reference_loss=2.8698,
        required_transformer_step=300,
    )

    # Disjoint hybrid/baseline steps → memory_metrics is None.
    assert result["hz0b_tracking"]["memory_metrics"] is None
    assert len(result["hz0b_tracking"]["tracked_metrics"]) > 0
    # gate_three has matched_steps=[] → incomplete.
    assert result["gates"]["decode_gap_reduced"]["status"] == "incomplete"
    assert "shows_memory_task_advantage" not in result["gates"]


def test_hz0a_gate_does_not_include_memory_in_gating() -> None:
    """All-zero memory metrics must NOT prevent ready_to_continue_scaling."""
    scorecard = {
        "hybrid": {
            "50": {
                "step": 50,
                "loss": 3.0,
                "tokens_seen_estimate": 25600.0,
                "estimated_train_flops": 6.0 * 100.0 * 25600.0,
                "tokens_per_second": 100.0,
                "associative_recall_accuracy": 0.0,
                "overwrite_retrieval_accuracy": 0.0,
                "protected_memory_accuracy": 0.0,
                "multi_anchor_retrieval_accuracy": 0.0,
                "multi_anchor_anchor_set_accuracy": 0.0,
                "recall_distance_32_accuracy": 0.0,
                "recall_distance_64_accuracy": 0.0,
                "recall_distance_128_accuracy": 0.0,
                "recall_distance_256_accuracy": 0.0,
            },
            "150": {
                "step": 150,
                "loss": 2.5,
                "tokens_seen_estimate": 76800.0,
                "estimated_train_flops": 6.0 * 100.0 * 76800.0,
                "tokens_per_second": 110.0,
                "associative_recall_accuracy": 0.0,
                "overwrite_retrieval_accuracy": 0.0,
                "protected_memory_accuracy": 0.0,
                "multi_anchor_retrieval_accuracy": 0.0,
                "multi_anchor_anchor_set_accuracy": 0.0,
                "recall_distance_32_accuracy": 0.0,
                "recall_distance_64_accuracy": 0.0,
                "recall_distance_128_accuracy": 0.0,
                "recall_distance_256_accuracy": 0.0,
            },
        },
        "baseline": {
            "50": {
                "step": 50,
                "loss": 3.3,
                "tokens_seen_estimate": 25600.0,
                "estimated_train_flops": 6.0 * 100.0 * 25600.0,
                "tokens_per_second": 180.0,
                "associative_recall_accuracy": 0.0,
                "overwrite_retrieval_accuracy": 0.0,
                "protected_memory_accuracy": 0.0,
                "multi_anchor_retrieval_accuracy": 0.0,
                "multi_anchor_anchor_set_accuracy": 0.0,
                "recall_distance_32_accuracy": 0.0,
                "recall_distance_64_accuracy": 0.0,
                "recall_distance_128_accuracy": 0.0,
                "recall_distance_256_accuracy": 0.0,
            },
            "150": {
                "step": 150,
                "loss": 2.85,
                "tokens_seen_estimate": 76800.0,
                "estimated_train_flops": 6.0 * 100.0 * 76800.0,
                "tokens_per_second": 190.0,
                "associative_recall_accuracy": 0.0,
                "overwrite_retrieval_accuracy": 0.0,
                "protected_memory_accuracy": 0.0,
                "multi_anchor_retrieval_accuracy": 0.0,
                "multi_anchor_anchor_set_accuracy": 0.0,
                "recall_distance_32_accuracy": 0.0,
                "recall_distance_64_accuracy": 0.0,
                "recall_distance_128_accuracy": 0.0,
                "recall_distance_256_accuracy": 0.0,
            },
        },
    }
    reference_manifest = {
        "checkpoint_step": 100,
        "effective_tokens_per_optimizer_update": 512,
        "parameter_count": 36_073_344,
    }

    # All-zero memory, all-zero hybrid advantage, but pass on gates 1 and 3 at
    # step 150 → ready_to_continue_scaling must still flip True.
    result = evaluate_hz0a_gates(
        scorecard=scorecard,
        reference_manifest=reference_manifest,
        reference_loss=2.8698,
        required_transformer_step=300,
    )

    assert result["gates"]["beats_36m_at_fair_tokens_per_param"]["status"] == "pass"
    assert result["gates"]["decode_gap_reduced"]["status"] == "pass"
    # Memory metrics are zero across the board. The HZ-0B tracking block still
    # surfaces the best matched (step, metric) pair; all-zero candidates are
    # valid, but they must NOT block ready_to_continue_scaling.
    memory_metrics = result["hz0b_tracking"]["memory_metrics"]
    assert memory_metrics is not None
    # Tie-break: smallest common step + first metric in HZ0B_MEMORY_METRICS.
    assert memory_metrics["step"] == 50
    assert memory_metrics["metric"] == "associative_recall_accuracy"
    assert memory_metrics["hybrid_value"] == 0.0
    assert memory_metrics["baseline_value"] == 0.0
    assert memory_metrics["advantage"] == 0.0
    assert result["ready_to_continue_scaling"] is True
    assert "shows_memory_task_advantage" not in result["gates"]
