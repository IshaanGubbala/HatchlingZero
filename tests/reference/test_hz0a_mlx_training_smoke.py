import json
import subprocess
import sys


def test_mlx_native_forward_training_smoke():
    result = subprocess.run(
        [sys.executable, "scripts/hz0a_mlx_training_smoke.py", "--steps", "2"],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["native_forward"] is True
    assert report["max_parameter_delta"] > 0
    assert len(report["metrics"]) == 2
    assert all(metric["loss"] == metric["loss"] for metric in report["metrics"])


def test_mlx_native_optimizer_replay_is_deterministic():
    command = [sys.executable, "scripts/hz0a_mlx_training_smoke.py", "--steps", "10", "--validation-interval", "5", "--seed", "11"]
    first = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    second = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    assert first["metrics"] == second["metrics"]
    assert first["max_parameter_delta"] == second["max_parameter_delta"]
    assert first["final_validation_loss"] < first["metrics"][0]["loss"]
    assert all(metric["gradient_norm"] == metric["gradient_norm"] and metric["update_norm"] > 0 for metric in first["metrics"])


def test_mlx_native_110m_one_step_smoke():
    result = subprocess.run(
        [sys.executable, "scripts/hz0a_mlx_training_smoke.py", "--steps", "1", "--seed", "7", "--vocab-size", "256", "--dim", "576", "--layers", "22", "--heads", "18", "--d-ff", "1728", "--sequence-length", "128", "--attention-every", "4", "--learning-rate", "0.0002"],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["native_forward"] is True
    assert report["parameter_count"] == 112150656
    assert report["metrics"][0]["loss"] == report["metrics"][0]["loss"]
    assert report["max_parameter_delta"] > 0
