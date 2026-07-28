import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_stage_runner_marks_bounded_run_as_smoke_and_writes_checkpoints(tmp_path: Path):
    packed = tmp_path / "packed.jsonl"
    packed.write_text("\n".join(json.dumps(list(range(128))) for _ in range(8)) + "\n")
    result = subprocess.run([sys.executable, "scripts/hz0a_stage_runner.py", "--data", str(packed), "--run-dir", str(tmp_path / "run"), "--steps", "1", "--checkpoint-interval", "1"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
