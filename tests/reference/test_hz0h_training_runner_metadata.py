from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


COMMON = {"device": "cpu", "hardware_id": "CPU", "effective_batch_tokens": 8,
          "compile_step": False, "compile_mode": None, "fused_optimizer": False}


def _run(command: list[str], path: Path) -> dict:
    subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(path.read_text())


def test_dense_bdh_runner_reports_fair_execution_metadata(tmp_path: Path):
    run_dir = tmp_path / "bdh"
    report = _run([sys.executable, "scripts/hz0h_stage2_runner_bdh.py", "--data", "data/packed/hz0h_bytes_25m_train.jsonl", "--validation-data", "data/packed/hz0h_bytes_25m_val.jsonl", "--run-dir", str(run_dir), "--target-tokens", "16", "--batch-size", "1", "--validation-batch-size", "1", "--sequence-length", "8", "--n-embd", "16", "--n-layer", "2", "--n-head", "2", "--mlp-internal-dim-multiplier", "4", "--checkpoint-interval", "2", "--validation-interval", "2", "--device", "cpu", "--dtype", "float32", "--warmup-steps", "0"], run_dir / "bdh_stage2.json")
    assert {key: report[key] for key in COMMON} == COMMON


def test_transformer_runner_reports_fair_execution_metadata(tmp_path: Path):
    run_dir = tmp_path / "transformer"
    report = _run([sys.executable, "scripts/hz0a_torch_stage2_runner.py", "--architecture", "transformer", "--rope", "--data", "data/packed/hz0h_bytes_25m_train.jsonl", "--validation-data", "data/packed/hz0h_bytes_25m_val.jsonl", "--run-dir", str(run_dir), "--target-tokens", "16", "--batch-size", "1", "--validation-batch-size", "1", "--sequence-length", "8", "--dim", "16", "--layers", "2", "--heads", "2", "--d-ff", "32", "--vocab-size", "256", "--checkpoint-interval", "2", "--validation-interval", "2", "--device", "cpu", "--dtype", "float32", "--warmup-steps", "0"], run_dir / "torch_stage2.json")
    assert {key: report[key] for key in COMMON} == COMMON


def test_transformer_runner_rejects_fused_optimizer_off_cuda(tmp_path: Path):
    command = [sys.executable, "scripts/hz0a_torch_stage2_runner.py", "--data", "x", "--validation-data", "y", "--run-dir", str(tmp_path / "run"), "--fused-optimizer", "--device", "cpu"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires --device cuda" in result.stderr
