import json
import subprocess
import sys
from pathlib import Path


def test_native_model_parity_report_is_machine_readable():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "scripts/hz0a_native_model_parity.py"], cwd=root, check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["finite"]
    assert report["max_absolute_output_error"] < 1e-3
    assert report["max_absolute_gradient_error"] < 5e-3
    assert report["parameter_update_max_absolute_error"] < 5e-3
    assert len(report["per_parameter_gradient_errors"]) > 10
