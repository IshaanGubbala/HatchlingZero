import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[2]


def test_tiny_checkpoint_audit_accepts_valid_and_rejects_corruption(tmp_path):
    run_dir = tmp_path / "run"
    subprocess.run([sys.executable, "scripts/hz0a_tiny_training_comparison.py", "--run-dir", str(run_dir), "--steps", "3"], cwd=ROOT, check=True, capture_output=True, text=True)
    auditor = [sys.executable, "scripts/hz0a_audit_tiny_checkpoint.py", "--checkpoint", str(run_dir / "hybrid.pt")]
    valid = subprocess.run(auditor, cwd=ROOT, check=True, capture_output=True, text=True)
    report = json.loads(valid.stdout)
    assert report["valid"] is True
    assert report["step"] == report["metric_count"] == 3
    assert report["parameter_count"] > 1
    corrupt = tmp_path / "corrupt.pt"
    payload = torch.load(run_dir / "hybrid.pt", weights_only=False)
    first = next(value for value in payload["model"].values() if value.is_floating_point())
    first.view(-1)[0] = float("inf")
    shutil.copyfile(run_dir / "hybrid.pt", corrupt)
    torch.save(payload, corrupt)
    failed = subprocess.run([*auditor[:-1], str(corrupt)], cwd=ROOT, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "non-finite" in (failed.stdout + failed.stderr)
