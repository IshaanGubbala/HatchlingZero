from __future__ import annotations

from scripts.hz0h_training_target_gate import evaluate


def _report(params=100, tokens=1000, dtype="bfloat16", throughput=130.0, seconds=7.0, ram=700, **overrides):
    report = {"parameter_count": params, "target_tokens": tokens, "tokens_seen": tokens, "dtype": dtype, "tokens_per_second": throughput, "training_seconds": seconds, "peak_memory_bytes": ram,
              "device": "cuda", "hardware_id": "RTX 3060", "effective_batch_tokens": 3072,
              "compile_step": False, "compile_mode": None, "fused_optimizer": False}
    report.update(overrides)
    return report


def test_training_gate_requires_both_speed_and_ram():
    result = evaluate(_report(), _report(100, throughput=100.0, seconds=10.0, ram=1000))
    assert result["speed_gate"] and result["ram_gate"]
    assert not result["claim_eligible"]


def test_training_gate_rejects_memory_or_speed_miss():
    result = evaluate(_report(100, throughput=110.0, seconds=9.1, ram=700), _report(100, throughput=100.0, seconds=10.0, ram=1000))
    assert not result["speed_gate"]
    assert result["ram_gate"]


def test_training_gate_rejects_token_mismatch():
    result = evaluate(_report(), _report(100, tokens=900, throughput=100.0, seconds=10.0, ram=1000))
    assert not result["speed_gate"] and not result["ram_gate"]


def test_training_gate_rejects_different_execution_conditions():
    transformer = _report(100, throughput=100.0, seconds=10.0, ram=1000)
    for field, value in (("hardware_id", "RTX 4090"), ("effective_batch_tokens", 1024), ("compile_step", True), ("fused_optimizer", True)):
        result = evaluate(_report(**{field: value}), transformer)
        assert not result["speed_gate"] and not result["ram_gate"]
        assert result["execution_conditions"][f"{field}_equal"] is False


def test_training_gate_rejects_underparameterized_candidate():
    result = evaluate(_report(params=98), _report(params=100, throughput=100.0, seconds=10.0, ram=1000))
    assert not result["speed_gate"] and not result["ram_gate"]
