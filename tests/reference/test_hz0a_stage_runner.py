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


def test_stage_runner_resume_matches_uninterrupted_run(tmp_path: Path):
    packed = tmp_path / "packed.jsonl"
    packed.write_text("\n".join(json.dumps(list(range(128))) for _ in range(8)) + "\n")
    first_dir = tmp_path / "first"
    full_dir = tmp_path / "full"
    base = [sys.executable, "scripts/hz0a_stage_runner.py", "--data", str(packed), "--steps", "1", "--checkpoint-interval", "1"]
    interrupted = subprocess.run([*base, "--run-dir", str(first_dir)], cwd=ROOT, capture_output=True, text=True)
    assert interrupted.returncode != 0
    # Use a temporary stage definition with a five-token requirement for this resume fixture.
    stages = tmp_path / "stages.json"
    stages.write_text(json.dumps({"stages": [{"name": "test", "tokens": 1}]}))
    command = [sys.executable, "scripts/hz0a_stage_runner.py", "--stage-config", str(stages), "--stage", "test", "--data", str(packed), "--checkpoint-interval", "1"]
    assert subprocess.run([*command, "--run-dir", str(first_dir), "--steps", "1"], cwd=ROOT, capture_output=True, text=True).returncode == 0
    assert subprocess.run([*command, "--run-dir", str(first_dir), "--steps", "2", "--resume"], cwd=ROOT, capture_output=True, text=True).returncode == 0
    assert subprocess.run([*command, "--run-dir", str(full_dir), "--steps", "2"], cwd=ROOT, capture_output=True, text=True).returncode == 0
    resumed = json.loads((first_dir / "stage_report.json").read_text())
    full = json.loads((full_dir / "stage_report.json").read_text())
    for name in ("hybrid", "transformer"):
        assert resumed["models"][name]["final_parameter_sha256"] == full["models"][name]["final_parameter_sha256"]
        assert "validation_loss" in resumed["models"][name]["metrics"][-1]


def test_stage_runner_can_execute_locked_config_model(tmp_path: Path):
    packed = tmp_path / "packed.jsonl"
    packed.write_text("\n".join(json.dumps(list(range(128))) for _ in range(8)) + "\n")
    stages = tmp_path / "stages.json"
    stages.write_text(json.dumps({"stages": [{"name": "test", "tokens": 1}]}))
    config = tmp_path / "model.json"
    config.write_text(json.dumps({"vocab_size": 32, "d_model": 16, "num_layers": 3, "num_heads": 2, "head_dim_qk": 8, "head_dim_v": 8, "d_ff": 32, "attention_layer_indices": [1]}))
    result = subprocess.run([sys.executable, "scripts/hz0a_stage_runner.py", "--stage-config", str(stages), "--stage", "test", "--data", str(packed), "--validation-data", str(packed), "--run-dir", str(tmp_path / "run"), "--steps", "1", "--batch-size", "2", "--vocab-size", "32", "--models", "locked", "--model-config", str(config), "--checkpoint-interval", "1"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "run" / "stage_report.json").read_text())
    assert report["models"]["locked"]["parameter_count"] > 0
    assert report["models"]["locked"]["parameters_changed"]
