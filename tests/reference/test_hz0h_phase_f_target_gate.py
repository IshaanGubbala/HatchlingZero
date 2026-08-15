from __future__ import annotations

from scripts.hz0h_phase_f_target_gate import evaluate


def _report(candidate_ram: int, candidate_speed: float) -> dict:
    return {
        "bdh_parameter_count": 100,
        "vb_parameter_count": 100,
        "transformer_parameter_count": 100,
        "by_context_length": {"1024": {
            "vb_decode_streaming_state_speed_mode": {"peak_memory_bytes": candidate_ram, "tokens_per_second": candidate_speed},
            "transformer_decode_kv_cache": {"peak_memory_bytes": 1000, "tokens_per_second": 100.0},
        }},
    }


def test_target_gate_passes_only_both_execution_thresholds():
    result = evaluate(_report(600, 130.0), "1024", "vb_decode_streaming_state_speed_mode")
    assert result["ram_gate"] and result["speed_gate"]
    assert not result["claim_eligible"]


def test_target_gate_rejects_single_metric_win():
    result = evaluate(_report(600, 120.0), "1024", "vb_decode_streaming_state_speed_mode")
    assert result["ram_gate"] and not result["speed_gate"]


def test_target_gate_rejects_parameter_mismatch():
    report = _report(600, 130.0)
    report["vb_parameter_count"] = 102
    result = evaluate(report, "1024", "vb_decode_streaming_state_speed_mode")
    assert not result["ram_gate"] and not result["speed_gate"]
