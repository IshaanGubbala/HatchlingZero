import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_checkpoint_evaluator_reports_shared_quality_metrics(tmp_path: Path):
    packed = tmp_path / "packed.jsonl"
    packed.write_text("\n".join(json.dumps(list(range(128))) for _ in range(8)) + "\n")
    stages = tmp_path / "stages.json"
    stages.write_text(json.dumps({"stages": [{"name": "test", "tokens": 1}]}))
    run = tmp_path / "run"
    command = [sys.executable, "scripts/hz0a_stage_runner.py", "--stage-config", str(stages), "--stage", "test", "--data", str(packed), "--run-dir", str(run), "--steps", "1", "--checkpoint-interval", "1"]
    assert subprocess.run(command, cwd=ROOT, capture_output=True, text=True).returncode == 0
    output = tmp_path / "evaluation.json"
    result = subprocess.run([sys.executable, "scripts/hz0a_evaluate_checkpoints.py", "--hybrid-checkpoint", str(run / "hybrid.pt"), "--transformer-checkpoint", str(run / "transformer.pt"), "--data", str(packed), "--batches", "1", "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert set(report["models"]) == {"hybrid", "transformer"}
    assert all(report["models"][name]["tokens"] > 0 and report["models"][name]["perplexity"] > 0 for name in report["models"])
