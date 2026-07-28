from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from restart.hz0a_harness import DeterministicHarness, HarnessConfig, audit_checkpoint_payload  # noqa: E402


def make_config(tmp_path: Path) -> HarnessConfig:
    run_dir = tmp_path / "run"
    return HarnessConfig(
        run_name="test",
        packed_data_path="data/packed/train_packed.json",
        sequence_length=128,
        microbatch_size=2,
        grad_accum_steps=4,
        max_optimizer_steps=3,
        validation_interval=1,
        checkpoint_interval=1,
        learning_rate=1e-4,
        seed=7,
    )


def test_effective_batch_tokens_matches_historical_shape() -> None:
    cfg = HarnessConfig(
        run_name="historical-shape",
        packed_data_path="data/packed/train_packed.json",
        sequence_length=256,
        microbatch_size=2,
        grad_accum_steps=4,
        max_optimizer_steps=1,
        validation_interval=1,
        checkpoint_interval=1,
        learning_rate=1e-4,
        seed=7,
    )
    assert cfg.effective_batch_tokens == 2048


def test_harness_tracks_token_accounting_and_snapshots(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    harness = DeterministicHarness(cfg, tmp_path / "run")
    harness.run()
    assert harness.state.optimizer_step == 3
    assert harness.state.microbatch_count == 12
    assert harness.state.tokens_seen == 12 * 2 * 128
    assert harness.state.effective_batch_tokens == 2 * 128 * 4
    snapshot = tmp_path / "run" / "config.snapshot.json"
    assert snapshot.exists()
    payload = json.loads(snapshot.read_text())
    assert payload["effective_batch_tokens"] == 1024
    assert payload["model_shape"]["vocab_size"] == cfg.model_vocab_size
    assert payload["model_shape"]["num_layers"] == cfg.model_num_layers
    assert harness.state.model_logit_scale != 1.0
    assert harness.state.model_param == harness.state.model_logit_scale
    assert harness.state.accumulated_scale_grad == 0.0
    assert len(harness.validation_history) == 3
    assert all("model_logit_scale" in metric for metric in harness.validation_history)
    assert all(record.scale_grad != 0.0 for record in harness.records)


def test_harness_real_loss_matches_direct_reference_computation(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    harness = DeterministicHarness(cfg, tmp_path / "run")

    batch, _, _ = harness.dataset.get_microbatch(0, cfg.microbatch_size)
    token_ids, targets = harness._batch_inputs_and_targets(batch)
    logits, _ = harness.model(token_ids)
    expected_loss, expected_scale_grad = harness._cross_entropy_with_scale_grad(logits, targets)

    record = harness.run_microbatch()

    assert record.loss == expected_loss
    assert record.scale_grad == expected_scale_grad
    assert record.gradient_norm == abs(expected_scale_grad)
    assert harness.state.last_loss == expected_loss
    assert harness.state.accumulated_scale_grad == expected_scale_grad


def test_harness_resume_is_exact(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)

    interrupted = DeterministicHarness(cfg, tmp_path / "interrupted")
    interrupted.run(stop_after_microbatches=5)
    ckpt = interrupted.save_checkpoint("resume_point")

    resumed = DeterministicHarness(cfg, tmp_path / "resumed")
    resumed.load_checkpoint(ckpt)
    resumed.run()

    full = DeterministicHarness(cfg, tmp_path / "full")
    full.run()

    assert as_jsonable(resumed) == as_jsonable(full)


def test_scheduler_and_separate_validation_split_resume_exactly(tmp_path: Path) -> None:
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps([[7] * 128, [9] * 128]))
    cfg = HarnessConfig(**{**make_config(tmp_path).__dict__, "scheduler": "cosine", "warmup_optimizer_steps": 1, "validation_packed_data_path": str(validation_path)})
    harness = DeterministicHarness(cfg, tmp_path / "run")
    harness.run()
    assert harness.state.scheduler_step == harness.state.optimizer_step == 3
    assert harness.state.current_learning_rate >= 0
    assert [item["learning_rate"] for item in harness.validation_history] == sorted((item["learning_rate"] for item in harness.validation_history), reverse=True)


def test_checkpoint_audit_validates_accounting_and_finite_values(tmp_path: Path) -> None:
    harness = DeterministicHarness(make_config(tmp_path), tmp_path / "run")
    harness.run(stop_after_microbatches=4)
    payload = json.loads(harness.save_checkpoint("audit").read_text())

    result = audit_checkpoint_payload(payload)

    assert result["finite"] is True
    assert result["record_count"] == 4
    assert result["tokens_seen"] == 4 * 2 * 128


def test_non_finite_logits_are_refused(tmp_path: Path) -> None:
    harness = DeterministicHarness(make_config(tmp_path), tmp_path / "run")
    batch, _, _ = harness.dataset.get_microbatch(0, harness.config.microbatch_size)
    token_ids, targets = harness._batch_inputs_and_targets(batch)
    logits, _ = harness.model(token_ids)
    logits[0, 0, 0] = np.inf

    try:
        harness._cross_entropy_with_scale_grad(logits, targets)
    except FloatingPointError as exc:
        assert "model logits" in str(exc)
    else:
        raise AssertionError("non-finite logits must be refused")


def as_jsonable(harness: DeterministicHarness) -> dict:
    return {
        "state": harness.state.__dict__,
        "records": [r.__dict__ for r in harness.records],
        "validation_history": harness.validation_history,
    }
