from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_blocksparse_real_corpus_runner_smoke(tmp_path: Path):
    run_dir = tmp_path / "run"
    command = [
        sys.executable, "scripts/hz0h_blocksparse_train.py",
        "--data", "data/packed/hz0h_bytes_25m_train.jsonl",
        "--validation-data", "data/packed/hz0h_bytes_25m_val.jsonl",
        "--run-dir", str(run_dir), "--target-tokens", "16", "--batch-size", "1",
        "--validation-batch-size", "1", "--sequence-length", "8", "--n-embd", "16",
        "--n-layer", "2", "--n-head", "2", "--mlp-internal-dim-multiplier", "4",
        "--block-size", "4", "--active-fraction", "0.5", "--checkpoint-interval", "2",
        "--validation-interval", "2", "--device", "cpu", "--dtype", "float32",
        "--warmup-steps", "0",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads((run_dir / "block_bdh_training.json").read_text())
    assert report["architecture"] == "block_bdh_derivative"
    assert report["exact_bdh"] is False
    assert report["claim_eligible"] is False
    assert report["compile_step"] is False
    assert report["effective_batch_tokens"] == 8
    assert report["hardware_id"] == "CPU"
    assert report["budget_complete"] is True
    assert report["best_validation_loss"] > 0
    assert report["metrics"][-1]["active_block_count"] == 4
    assert (run_dir / "block_bdh_checkpoint.pt").exists()


def test_blocksparse_runner_rejects_unfair_compile():
    command = [sys.executable, "scripts/hz0h_blocksparse_train.py", "--data", "x", "--validation-data", "y", "--run-dir", "/tmp/nope", "--compile-step"]
    outcome = subprocess.run(command, capture_output=True, text=True)
    assert outcome.returncode != 0
    assert "dynamic BlockBDH routing" in outcome.stderr


def test_blocksparse_runner_rejects_nondividing_block_size():
    command = [sys.executable, "scripts/hz0h_blocksparse_train.py", "--data", "x", "--validation-data", "y", "--run-dir", "/tmp/nope", "--n-embd", "16", "--n-head", "2", "--mlp-internal-dim-multiplier", "4", "--block-size", "6"]
    outcome = subprocess.run(command, capture_output=True, text=True)
    assert outcome.returncode != 0
    assert "must divide latent width" in outcome.stderr
