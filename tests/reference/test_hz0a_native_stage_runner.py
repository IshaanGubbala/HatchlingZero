import json
import subprocess
import sys


def test_native_stage_runner_exposes_resumable_stage_contract():
    result = subprocess.run([sys.executable, "scripts/hz0a_native_stage_runner.py", "--help"], check=True, capture_output=True, text=True)
    assert "--resume" in result.stdout
    assert "--target-tokens" in result.stdout
    assert "--checkpoint-interval" in result.stdout


def _write_jsonl(path, sequences):
    path.write_text("\n".join(json.dumps(sequence) for sequence in sequences) + "\n", encoding="utf-8")


def test_native_stage_runner_effective_batch_tokens_matches_historical_shape(tmp_path):
    """A7 plan requirement: batch 2 x sequence 256 x accumulation 4 = 2,048
    tokens per optimizer step, proven for the harness actually in use
    (scripts/hz0a_native_stage_runner.py), not just the older
    restart/hz0a_harness.py config object."""
    sequence_length = 256
    data = tmp_path / "train.jsonl"
    validation = tmp_path / "val.jsonl"
    _write_jsonl(data, [[1] * sequence_length])
    _write_jsonl(validation, [[1] * sequence_length])
    run_dir = tmp_path / "run"
    subprocess.run(
        [
            sys.executable, "scripts/hz0a_native_stage_runner.py",
            "--data", str(data), "--validation-data", str(validation),
            "--run-dir", str(run_dir), "--target-tokens", "1",
            "--batch-size", "2", "--sequence-length", str(sequence_length),
            "--chunk-length", "256", "--truncate-backward", "--gradient-accumulation-chunks", "4",
            "--vocab-size", "8", "--dim", "8", "--layers", "1", "--heads", "2", "--d-ff", "8",
            "--validation-batch-size", "1",
        ],
        check=True, capture_output=True, text=True,
    )
    config_snapshot = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    assert config_snapshot["effective_batch_tokens"] == 2048


def test_native_stage_runner_resume_preserves_microbatch_and_epoch_accounting(tmp_path):
    """Deterministic short run resumes exactly: microbatch_count and
    epoch_or_data_pass must continue monotonically across a --resume
    boundary, never reset or double-count (A7 exit gate)."""
    sequence_length = 16
    data = tmp_path / "train.jsonl"
    validation = tmp_path / "val.jsonl"
    sequences = [[index % 8 for index in range(sequence_length)] for _ in range(3)]
    _write_jsonl(data, sequences)
    _write_jsonl(validation, sequences)
    run_dir = tmp_path / "run"
    base_args = [
        sys.executable, "scripts/hz0a_native_stage_runner.py",
        "--data", str(data), "--validation-data", str(validation),
        "--run-dir", str(run_dir), "--batch-size", "2",
        "--checkpoint-interval", "1", "--validation-interval", "1",
        "--sequence-length", str(sequence_length),
        "--vocab-size", "8", "--dim", "8", "--layers", "1", "--heads", "2", "--d-ff", "8",
        "--validation-batch-size", "2", "--architecture", "transformer",
    ]
    first = subprocess.run(base_args + ["--target-tokens", "96"], check=True, capture_output=True, text=True)
    first_report = json.loads(first.stdout)
    assert first_report["microbatch_count"] == 3
    assert first_report["epoch_or_data_pass"] == 1  # 3 sequences, batch 2 -> wraps once by step 2

    second = subprocess.run(base_args + ["--target-tokens", "192", "--resume"], check=True, capture_output=True, text=True)
    second_report = json.loads(second.stdout)
    assert second_report["microbatch_count"] == 6  # continues 4,5,6 -- not reset to 1
    assert second_report["epoch_or_data_pass"] >= first_report["epoch_or_data_pass"]
    assert len(second_report["metrics"]) == 6  # restored history (1-3) plus new steps (4-6), never replayed
    assert second_report["metrics"][-1]["step"] == 6

    audit = subprocess.run(
        [sys.executable, "scripts/hz0a_audit_native_checkpoint.py", str(run_dir / "native_metal_checkpoint"), "--required-tokens", "192"],
        check=True, capture_output=True, text=True,
    )
    audit_report = json.loads(audit.stdout)
    assert audit_report["budget_complete"] is True
    assert audit_report["checkpoint_tensors_finite"] is True
    assert audit_report["model_parameter_sha256"] == second_report["final_parameter_sha256"]
