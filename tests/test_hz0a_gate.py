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
        memory_metric_threshold=0.1,
        memory_advantage_delta=0.05,
    )

    assert result["gates"]["beats_36m_at_fair_tokens_per_param"]["status"] == "pass"
    assert result["gates"]["maintains_transformer_advantage_through_horizon"]["status"] == "pass"
    assert result["gates"]["decode_gap_reduced"]["status"] == "pass"
    assert result["gates"]["shows_memory_task_advantage"]["status"] == "fail"
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
    assert result["gates"]["shows_memory_task_advantage"]["status"] == "fail"
