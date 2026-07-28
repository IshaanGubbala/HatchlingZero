from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def run_replay(output: Path) -> dict:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/hz0a_optimizer_replay.py"), "--steps", "100", "--output", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_a9_replay_is_exact_and_stable(tmp_path: Path) -> None:
    first = run_replay(tmp_path / "first.json")
    second = run_replay(tmp_path / "second.json")

    assert first == second
    assert first["steps"] == 100
    assert first["learning_rate"] == 1e-4
    assert first["stable_finite"] is True
    assert first["batch_order"][:8] == list(range(8))
    assert first["metrics"][0]["loss"] > first["metrics"][-1]["loss"]
    assert first["final_parameter_sha256"]
