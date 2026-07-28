from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from restart.hz0a_harness import DeterministicHarness, HarnessConfig  # noqa: E402


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


def as_jsonable(harness: DeterministicHarness) -> dict:
    return {
        "state": harness.state.__dict__,
        "records": [r.__dict__ for r in harness.records],
        "validation_history": harness.validation_history,
    }
