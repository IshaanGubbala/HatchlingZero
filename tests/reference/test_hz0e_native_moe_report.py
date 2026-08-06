import json
import subprocess
import sys
from pathlib import Path


def test_native_moe_report_is_machine_readable_and_finite():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "scripts/hz0e_native_moe_report.py", "--steps", "2"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["steps"] == 2
    assert report["tokens"] == 16
    assert report["finite"]
    assert report["gradient_finite"]
    assert len(report["parameter_fingerprint"]) == 64
    assert all(count >= 0 for count in report["overflow_counts"])
