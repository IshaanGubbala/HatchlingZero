import json
import subprocess
import sys
from pathlib import Path


def test_native_metal_replay_matches_reference_mlx():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "scripts/hz0a_mlx_native_replay.py", "--steps", "100"], cwd=root, check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["steps"] == 100
    assert report["finite"]
    assert report["max_loss_difference"] < 0.01
    assert report["max_gradient_error"] < 0.02
    assert report["max_update_error"] < 0.005
