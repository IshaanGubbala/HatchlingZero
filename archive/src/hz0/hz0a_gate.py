from __future__ import annotations

import json
from pathlib import Path
from typing import Any


Status = str


# Set of metrics the staged plan tracks as HZ-0B memory evidence. Kept here
# (not in tests) so the gate logic and the scorecard consumers agree.
HZ0B_MEMORY_METRICS: tuple[str, ...] = (
    "associative_recall_accuracy",
    "overwrite_retrieval_accuracy",
    "protected_memory_accuracy",
    "multi_anchor_retrieval_accuracy",
    "multi_anchor_anchor_set_accuracy",
    "recall_distance_32_accuracy",
    "recall_distance_64_accuracy",
    "recall_distance_128_accuracy",
    "recall_distance_256_accuracy",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sorted_step_metrics(step_metrics: dict[str, Any]) -> list[tuple[int, dict[str, float]]]:
    items: list[tuple[int, dict[str, float]]] = []
    for key, value in step_metrics.items():
        items.append((int(key), value))
    return sorted(items, key=lambda item: item[0])


def _tokens_per_parameter(tokens_seen: float, parameter_count: float) -> float:
    if parameter_count <= 0:
        return 0.0
    return float(tokens_seen) / float(parameter_count)


def _metric_value(metrics: dict[str, float], name: str) -> float:
    return float(metrics.get(name, 0.0))


def _status_payload(status: Status, summary: str, **details: Any) -> dict[str, Any]:
    payload = {"status": status, "summary": summary}
    payload.update(details)
    return payload


def _summarise_hz0b_memory(scorecard: dict[str, Any]) -> dict[str, Any]:
    """Informational summary of the HZ-0B memory metrics from the scorecard.

    Memory-task advantage is NOT an HZ-0A gate. It is the HZ-0B scratchpad's
    real proof obligation. We still surface the best hybrid-vs-baseline memory
    advantage we have, but only as tracking — not as a blocking condition.
    Consumers can derive their own pass/fail by comparing ``hybrid_value`` and
    ``advantage`` against whatever threshold they care about.

    Return shape:
        {
            "memory_metrics": dict | None,
                # None when there are no matched hybrid/baseline steps.
                # Otherwise: {summary, step, metric, hybrid_value, baseline_value, advantage}.
            "tracked_metrics": list[str],
        }
    """
    hybrid_steps = _sorted_step_metrics(scorecard["hybrid"])
    baseline_steps = _sorted_step_metrics(scorecard["baseline"])
    hybrid_by_step = {step: metrics for step, metrics in hybrid_steps}
    baseline_by_step = {step: metrics for step, metrics in baseline_steps}
    common_steps = sorted(set(hybrid_by_step) & set(baseline_by_step))

    best_advantage: dict[str, Any] | None = None
    for step in common_steps:
        for metric_name in HZ0B_MEMORY_METRICS:
            hybrid_value = _metric_value(hybrid_by_step[step], metric_name)
            baseline_value = _metric_value(baseline_by_step[step], metric_name)
            advantage = hybrid_value - baseline_value
            candidate = {
                "step": step,
                "metric": metric_name,
                "hybrid_value": hybrid_value,
                "baseline_value": baseline_value,
                "advantage": advantage,
            }
            if best_advantage is None or candidate["advantage"] > best_advantage["advantage"]:
                best_advantage = candidate

    if best_advantage is None:
        memory_metrics_section: dict[str, Any] | None = None
    else:
        memory_metrics_section = {
            "summary": "Best matched hybrid-vs-baseline memory advantage observed across the scorecard.",
            **best_advantage,
        }

    return {
        "memory_metrics": memory_metrics_section,
        "tracked_metrics": list(HZ0B_MEMORY_METRICS),
    }


def evaluate_hz0a_gates(
    *,
    scorecard: dict[str, Any],
    reference_manifest: dict[str, Any],
    reference_loss: float,
    required_transformer_step: int = 300,
    min_loss_margin: float = 0.05,
    min_decode_ratio: float = 0.5,
) -> dict[str, Any]:
    hybrid_steps = _sorted_step_metrics(scorecard["hybrid"])
    baseline_steps = _sorted_step_metrics(scorecard["baseline"])
    hybrid_by_step = {step: metrics for step, metrics in hybrid_steps}
    baseline_by_step = {step: metrics for step, metrics in baseline_steps}
    common_steps = sorted(set(hybrid_by_step) & set(baseline_by_step))

    reference_tokens = float(reference_manifest["checkpoint_step"]) * float(
        reference_manifest["effective_tokens_per_optimizer_update"]
    )
    reference_params = float(reference_manifest["parameter_count"])
    reference_tokens_per_param = _tokens_per_parameter(reference_tokens, reference_params)

    hybrid_param_count = float(max(scorecard["hybrid"].values(), key=lambda item: float(item["step"]))["estimated_train_flops"])
    # Recover param count from FLOPs estimate = 6 * params * tokens_seen.
    latest_hybrid_metrics = hybrid_steps[-1][1]
    latest_hybrid_tokens = float(latest_hybrid_metrics["tokens_seen_estimate"])
    if latest_hybrid_tokens > 0:
        hybrid_param_count = float(latest_hybrid_metrics["estimated_train_flops"]) / (6.0 * latest_hybrid_tokens)
    else:
        hybrid_param_count = 0.0

    eligible_hybrid_steps: list[tuple[int, dict[str, float], float]] = []
    for step, metrics in hybrid_steps:
        tokens_per_param = _tokens_per_parameter(float(metrics["tokens_seen_estimate"]), hybrid_param_count)
        if tokens_per_param >= reference_tokens_per_param:
            eligible_hybrid_steps.append((step, metrics, tokens_per_param))

    if not eligible_hybrid_steps:
        gate_one = _status_payload(
            "incomplete",
            "No hybrid checkpoint has yet matched the 36M reference tokens-per-parameter budget.",
            reference_checkpoint_step=int(reference_manifest["checkpoint_step"]),
            reference_tokens_per_parameter=reference_tokens_per_param,
            max_hybrid_tokens_per_parameter=max(
                _tokens_per_parameter(float(metrics["tokens_seen_estimate"]), hybrid_param_count)
                for _, metrics in hybrid_steps
            ),
        )
    else:
        best_eligible_step, best_eligible_metrics, eligible_tokens_per_param = min(
            eligible_hybrid_steps,
            key=lambda item: float(item[1]["loss"]),
        )
        if float(best_eligible_metrics["loss"]) < reference_loss:
            gate_one = _status_payload(
                "pass",
                "A hybrid checkpoint beats the 36M reference after matching its tokens-per-parameter budget.",
                best_step=best_eligible_step,
                best_loss=float(best_eligible_metrics["loss"]),
                reference_loss=reference_loss,
                tokens_per_parameter=eligible_tokens_per_param,
            )
        else:
            gate_one = _status_payload(
                "fail",
                "The hybrid has reached the 36M reference tokens-per-parameter budget but has not beaten its loss yet.",
                best_step=best_eligible_step,
                best_loss=float(best_eligible_metrics["loss"]),
                reference_loss=reference_loss,
                tokens_per_parameter=eligible_tokens_per_param,
            )

    if not common_steps or max(common_steps) < required_transformer_step:
        gate_two = _status_payload(
            "incomplete",
            "Matched hybrid vs transformer continuation has not yet reached the required step horizon.",
            required_step=required_transformer_step,
            available_steps=common_steps,
        )
    else:
        checked_steps = [step for step in common_steps if step <= required_transformer_step]
        margins = {
            step: float(baseline_by_step[step]["loss"]) - float(hybrid_by_step[step]["loss"])
            for step in checked_steps
        }
        if all(margin >= min_loss_margin for margin in margins.values()):
            gate_two = _status_payload(
                "pass",
                "The hybrid keeps a clear matched-step loss advantage over the transformer through the required horizon.",
                checked_steps=checked_steps,
                min_margin=min(margins.values()),
                margins=margins,
            )
        else:
            gate_two = _status_payload(
                "fail",
                "The matched continuation does not maintain the required loss margin over the transformer.",
                checked_steps=checked_steps,
                min_margin=min(margins.values()),
                margins=margins,
            )

    decode_ratios = {
        step: _metric_value(hybrid_by_step[step], "tokens_per_second") / max(_metric_value(baseline_by_step[step], "tokens_per_second"), 1e-9)
        for step in common_steps
    }
    best_decode_ratio_step = max(decode_ratios, key=decode_ratios.get) if decode_ratios else None
    best_decode_ratio = decode_ratios[best_decode_ratio_step] if best_decode_ratio_step is not None else 0.0
    if best_decode_ratio_step is None:
        gate_three = _status_payload(
            "incomplete",
            "No matched checkpoints are available to compare decode speed.",
        )
    elif best_decode_ratio >= min_decode_ratio:
        gate_three = _status_payload(
            "pass",
            "The recurrent candidate has reduced the decode gap below the configured threshold.",
            best_step=best_decode_ratio_step,
            best_decode_ratio=best_decode_ratio,
            min_decode_ratio=min_decode_ratio,
            ratios=decode_ratios,
        )
    else:
        gate_three = _status_payload(
            "fail",
            "The recurrent candidate remains too slow relative to the matched transformer.",
            best_step=best_decode_ratio_step,
            best_decode_ratio=best_decode_ratio,
            min_decode_ratio=min_decode_ratio,
            ratios=decode_ratios,
        )

    hz0b_tracking = _summarise_hz0b_memory(scorecard)

    ready_to_continue_scaling = (
        gate_one["status"] == "pass"
        or gate_two["status"] == "pass"
        or gate_three["status"] == "pass"
    )
    return {
        "reference": {
            "checkpoint_step": int(reference_manifest["checkpoint_step"]),
            "parameter_count": int(reference_params),
            "tokens_per_optimizer_update": int(reference_manifest["effective_tokens_per_optimizer_update"]),
            "tokens_per_parameter": reference_tokens_per_param,
        },
        "candidate": {
            "available_steps": [step for step, _ in hybrid_steps],
            "matched_steps": common_steps,
            "estimated_parameter_count": hybrid_param_count,
        },
        "gates": {
            "beats_36m_at_fair_tokens_per_param": gate_one,
            "maintains_transformer_advantage_through_horizon": gate_two,
            "decode_gap_reduced": gate_three,
        },
        "hz0b_tracking": hz0b_tracking,
        "ready_to_continue_scaling": ready_to_continue_scaling,
    }


def evaluate_hz0a_gate_paths(
    *,
    scorecard_path: Path,
    reference_manifest_path: Path,
    reference_loss: float,
    required_transformer_step: int = 300,
    min_loss_margin: float = 0.05,
    min_decode_ratio: float = 0.5,
) -> dict[str, Any]:
    return evaluate_hz0a_gates(
        scorecard=_load_json(scorecard_path),
        reference_manifest=_load_json(reference_manifest_path),
        reference_loss=reference_loss,
        required_transformer_step=required_transformer_step,
        min_loss_margin=min_loss_margin,
        min_decode_ratio=min_decode_ratio,
    )
