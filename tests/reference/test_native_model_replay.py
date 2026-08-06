import json
from pathlib import Path
import subprocess
import sys


def test_native_model_replay_checkpoint_resume_is_exact(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "scripts/hz0a_native_model_replay.py", "--steps", "100", "--checkpoint", str(tmp_path / "native.json")], cwd=root, check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["steps"] == 100
    assert report["tokens"] == 800
    assert report["finite"]
    assert report["exact_resume"]


def test_native_moe_model_replay_checkpoint_resume_is_exact(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "scripts/hz0a_native_model_replay.py", "--steps", "20", "--moe", "--checkpoint", str(tmp_path / "moe.json")], cwd=root, check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["steps"] == 20
    assert report["tokens"] == 160
    assert report["moe"]
    assert report["finite"]
    assert report["exact_resume"]
