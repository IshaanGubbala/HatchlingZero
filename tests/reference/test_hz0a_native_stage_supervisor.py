import json
import subprocess
import sys
from pathlib import Path


def test_sequential_supervisor_completes_tiny_corrected_run(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    run_dir = tmp_path / "supervised"
    command = [
        sys.executable, "scripts/hz0a_native_stage_supervisor.py",
        "--data", "data/packed/stage1_10m_train.jsonl",
        "--validation-data", "data/packed/repro_1024_val.jsonl",
        "--run-dir", str(run_dir), "--target-tokens", "1024", "--batch-size", "1",
        "--checkpoint-interval", "2", "--validation-interval", "2", "--chunk-length", "128",
        "--vocab-size", "8192", "--dim", "32", "--layers", "1", "--heads", "2",
        "--d-ff", "64", "--mixer", "gdn2_fix", "--max-restarts", "3",
    ]
    subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    report = json.loads((run_dir / "native_metal.json").read_text())
    assert report["mixer"] == "gdn2_fix"
    assert report["budget_complete"] is True
    assert report["tokens_seen"] == 1024
