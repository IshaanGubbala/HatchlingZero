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
