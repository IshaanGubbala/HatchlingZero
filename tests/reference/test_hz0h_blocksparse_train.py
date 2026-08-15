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
    assert report["router_method"] == "activation"
    assert report["value_path"] == "vanilla"
    assert report["effective_batch_tokens"] == 8
    assert report["hardware_id"] == "CPU"
    assert report["budget_complete"] is True
    assert report["best_validation_loss"] > 0
    assert report["metrics"][-1]["active_block_count"] == 4
    assert len(report["metrics"][-1]["active_block_indices"]) == 4
    assert report["metrics"][0]["route_jaccard_previous"] is None
    assert report["route_summary"]["n_blocks"] == 8
    assert 0 < report["route_summary"]["selected_block_coverage"] <= 1
    assert report["route_summary"]["unique_route_sets"] >= 1
    assert (run_dir / "block_bdh_checkpoint.pt").exists()


def test_blocksparse_runner_compiles_selected_column_forward(tmp_path: Path):
    run_dir = tmp_path / "compiled"
    command = [sys.executable, "scripts/hz0h_blocksparse_train.py", "--data", "data/packed/hz0h_bytes_25m_train.jsonl", "--validation-data", "data/packed/hz0h_bytes_25m_val.jsonl", "--run-dir", str(run_dir), "--target-tokens", "8", "--batch-size", "1", "--validation-batch-size", "1", "--sequence-length", "8", "--n-embd", "16", "--n-layer", "1", "--n-head", "2", "--mlp-internal-dim-multiplier", "4", "--block-size", "4", "--active-fraction", "0.5", "--checkpoint-interval", "1", "--validation-interval", "1", "--device", "cpu", "--dtype", "float32", "--warmup-steps", "0", "--compile-step"]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    report = json.loads((run_dir / "block_bdh_training.json").read_text())
    assert report["compile_step"] is True
    assert report["compile_mode"] == "default"


def test_blocksparse_runner_rejects_nondividing_block_size():
    command = [sys.executable, "scripts/hz0h_blocksparse_train.py", "--data", "x", "--validation-data", "y", "--run-dir", "/tmp/nope", "--n-embd", "16", "--n-head", "2", "--mlp-internal-dim-multiplier", "4", "--block-size", "6"]
    outcome = subprocess.run(command, capture_output=True, text=True)
    assert outcome.returncode != 0
    assert "must divide latent width" in outcome.stderr


def test_blocksparse_runner_labels_direct_split_v_as_derivative(tmp_path: Path):
    run_dir = tmp_path / "direct_split_v"
    command = [sys.executable, "scripts/hz0h_blocksparse_train.py", "--data", "data/packed/hz0h_bytes_25m_train.jsonl", "--validation-data", "data/packed/hz0h_bytes_25m_val.jsonl", "--run-dir", str(run_dir), "--target-tokens", "8", "--batch-size", "1", "--validation-batch-size", "1", "--sequence-length", "8", "--n-embd", "16", "--n-layer", "1", "--n-head", "2", "--mlp-internal-dim-multiplier", "4", "--block-size", "4", "--active-fraction", "0.5", "--value-path", "direct_split_v", "--checkpoint-interval", "1", "--validation-interval", "1", "--device", "cpu", "--dtype", "float32", "--warmup-steps", "0"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads((run_dir / "block_bdh_training.json").read_text())
    assert report["architecture"] == "block_bdh_direct_split_v_derivative"
    assert report["value_path"] == "direct_split_v"
    assert report["exact_bdh"] is False
