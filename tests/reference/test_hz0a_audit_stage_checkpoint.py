import json
import subprocess
import sys
from pathlib import Path


def test_stage_checkpoint_auditor_reports_progress(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    import torch

    torch.save({"step": 2, "metrics": [{"step": 2, "validation_loss": 3.0}], "model": {"w": torch.ones(2)}, "dataset_cursor": {"offset": 4}, "model_parameter_sha256": "abc", "device": "cpu", "dtype": "torch.float32"}, checkpoint)
    result = subprocess.run([sys.executable, "scripts/hz0a_audit_stage_checkpoint.py", str(checkpoint), "--required-tokens", "10", "--batch-size", "1", "--sequence-length", "6"], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["tokens_seen"] == 10
    assert report["budget_complete"] is True
    assert report["model_tensors_finite"] is True
