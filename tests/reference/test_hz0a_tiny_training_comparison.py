from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def run(run_dir: Path, steps: int, resume: bool = False) -> dict:
    command = [sys.executable, "scripts/hz0a_tiny_training_comparison.py", "--run-dir", str(run_dir), "--steps", str(steps)]
    if resume:
        command.append("--resume")
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))


def test_tiny_models_train_on_shared_batches_and_resume_exactly(tmp_path: Path) -> None:
    first = run(tmp_path / "first", 4)
    repeated = run(tmp_path / "repeated", 4)
    assert first == repeated
    assert first["shared_effective_batch_tokens"] == 128
    for result in first["models"].values():
        assert result["steps"] == 4
        assert result["parameters_changed"] is True
        assert all(metric["gradient_norm"] > 0 for metric in result["metrics"])
        assert result["final_parameter_sha256"]

    interrupted = run(tmp_path / "resume", 2)
    resumed = run(tmp_path / "resume", 4, resume=True)
    assert resumed["models"] == first["models"]
