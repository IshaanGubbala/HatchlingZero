from __future__ import annotations

from scripts.hz0h_phase_f_target_gate import evaluate


def _report(candidate_ram: int, candidate_speed: float) -> dict:
    return {
        "bdh_parameter_count": 100,
        "vb_parameter_count": 100,
        "transformer_parameter_count": 100,
        "all_models_trained": True,
        "by_context_length": {"1024": {
            "vb_decode_streaming_state_speed_mode": {"peak_memory_bytes": candidate_ram, "tokens_per_second": candidate_speed},
            "transformer_decode_kv_cache": {"peak_memory_bytes": 1000, "tokens_per_second": 100.0},
        }},
    }


def test_target_gate_passes_only_both_execution_thresholds():
    result = evaluate(_report(600, 130.0), "1024", "vb_decode_streaming_state_speed_mode")
    assert result["ram_gate"] and result["speed_gate"] and result["target_evidence_gate"]
    assert not result["claim_eligible"]


def test_target_gate_rejects_single_metric_win():
    result = evaluate(_report(600, 120.0), "1024", "vb_decode_streaming_state_speed_mode")
    assert result["ram_gate"] and not result["speed_gate"]


def test_target_gate_rejects_parameter_mismatch():
    report = _report(600, 130.0)
    report["vb_parameter_count"] = 102
    result = evaluate(report, "1024", "vb_decode_streaming_state_speed_mode")
    assert not result["ram_gate"] and not result["speed_gate"]


def test_checkpoint_loader_requires_compatible_payload(tmp_path):
    import torch
    from scripts.hz0h_inference_benchmark import load_model_checkpoint

    source = torch.nn.Linear(3, 2)
    path = tmp_path / "wrapped.pt"
    torch.save({"model": source.state_dict()}, path)
    target = torch.nn.Linear(3, 2)
    meta = load_model_checkpoint(target, path)
    assert meta["trained_weights"] is True
    for a, b in zip(source.parameters(), target.parameters()):
        assert torch.equal(a, b)

    bad = tmp_path / "bad.pt"
    torch.save({"model": torch.nn.Linear(4, 2).state_dict()}, bad)
    try:
        load_model_checkpoint(torch.nn.Linear(3, 2), bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError("incompatible checkpoint was silently accepted")


def test_target_gate_rejects_underparameterized_candidate():
    report = _report(600, 130.0)
    report["vb_parameter_count"] = 98
    result = evaluate(report, "1024", "vb_decode_streaming_state_speed_mode")
    assert not result["parameter_match"]
    assert not result["ram_gate"] and not result["speed_gate"]


def test_target_evidence_requires_explicit_trained_checkpoint_provenance():
    report = _report(600, 130.0)
    del report["all_models_trained"]
    result = evaluate(report, "1024", "vb_decode_streaming_state_speed_mode")
    assert result["ram_gate"] and result["speed_gate"]
    assert result["trained_checkpoint_gate"] is False
    assert result["target_evidence_gate"] is False
