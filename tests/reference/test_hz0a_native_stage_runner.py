import subprocess
import sys


def test_native_stage_runner_exposes_resumable_stage_contract():
    result = subprocess.run([sys.executable, "scripts/hz0a_native_stage_runner.py", "--help"], check=True, capture_output=True, text=True)
    assert "--resume" in result.stdout
    assert "--target-tokens" in result.stdout
    assert "--checkpoint-interval" in result.stdout
