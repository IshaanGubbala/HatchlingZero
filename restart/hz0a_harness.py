from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import yaml


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@dataclass
class HarnessConfig:
    run_name: str
    packed_data_path: str
    sequence_length: int
    microbatch_size: int
    grad_accum_steps: int
    max_optimizer_steps: int
    validation_interval: int
    checkpoint_interval: int
    learning_rate: float
    seed: int

    @property
    def effective_batch_tokens(self) -> int:
        return self.microbatch_size * self.sequence_length * self.grad_accum_steps


@dataclass
class HarnessState:
    microbatch_count: int = 0
    optimizer_step: int = 0
    tokens_seen: int = 0
    effective_batch_tokens: int = 0
    epoch_or_data_pass: int = 0
    dataset_index: int = 0
    rng_seed: int = 0
    model_param: float = 0.0
    scheduler_step: int = 0
    last_loss: float | None = None
    peak_memory_bytes: int = 0
    gradient_norm: float = 0.0
    parameter_update_norm: float = 0.0


@dataclass
class MicrobatchRecord:
    microbatch_index: int
    optimizer_step_at_start: int
    dataset_index: int
    loss: float
    gradient_norm: float
    tokens: int


class PackedSequenceDataset:
    def __init__(self, packed_json_path: str | Path):
        self.path = Path(packed_json_path)
        self.sequences = json.loads(self.path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.sequences)

    def get_microbatch(self, start_index: int, microbatch_size: int) -> tuple[np.ndarray, int, int]:
        indices = [(start_index + i) % len(self.sequences) for i in range(microbatch_size)]
        wrapped = 1 if start_index + microbatch_size > len(self.sequences) else 0
        batch = np.array([self.sequences[i] for i in indices], dtype=np.int64)
        return batch, indices[-1] + 1, wrapped


class DeterministicHarness:
    def __init__(self, config: HarnessConfig, run_dir: str | Path):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = PackedSequenceDataset(config.packed_data_path)
        self.state = HarnessState(
            effective_batch_tokens=config.effective_batch_tokens,
            rng_seed=config.seed,
        )
        self.random = random.Random(config.seed)
        self.records: list[MicrobatchRecord] = []
        self.validation_history: list[dict[str, Any]] = []
        self.peak_memory_bytes = 0

    def snapshot_config(self) -> Path:
        payload = {
            "config": asdict(self.config),
            "packed_data_sha256": sha256_file(Path(self.config.packed_data_path)),
            "effective_batch_tokens": self.config.effective_batch_tokens,
        }
        path = self.run_dir / "config.snapshot.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _fake_loss(self, batch: np.ndarray) -> float:
        return float(np.mean(batch) / 1000.0 + self.state.optimizer_step * 1e-4)

    def _fake_gradient(self, loss: float) -> float:
        return float(abs(loss) * 0.5 + 0.01)

    def _fake_param_update(self, grad_norm: float) -> float:
        return float(self.config.learning_rate * grad_norm)

    def run_microbatch(self) -> MicrobatchRecord:
        batch, next_index, wrapped = self.dataset.get_microbatch(
            self.state.dataset_index,
            self.config.microbatch_size,
        )
        self.state.dataset_index = next_index % len(self.dataset)
        self.state.epoch_or_data_pass += wrapped

        loss = self._fake_loss(batch)
        grad_norm = self._fake_gradient(loss)
        tokens = self.config.microbatch_size * self.config.sequence_length
        self.state.microbatch_count += 1
        self.state.tokens_seen += tokens
        self.state.last_loss = loss
        self.state.gradient_norm = grad_norm
        self.peak_memory_bytes = max(self.peak_memory_bytes, int(batch.nbytes))
        self.state.peak_memory_bytes = self.peak_memory_bytes

        record = MicrobatchRecord(
            microbatch_index=self.state.microbatch_count,
            optimizer_step_at_start=self.state.optimizer_step,
            dataset_index=self.state.dataset_index,
            loss=loss,
            gradient_norm=grad_norm,
            tokens=tokens,
        )
        self.records.append(record)
        return record

    def maybe_step_optimizer(self) -> bool:
        if self.state.microbatch_count % self.config.grad_accum_steps != 0:
            return False
        update_norm = self._fake_param_update(self.state.gradient_norm)
        self.state.model_param -= update_norm
        self.state.parameter_update_norm = update_norm
        self.state.optimizer_step += 1
        self.state.scheduler_step += 1
        return True

    def validate(self) -> dict[str, Any]:
        metric = {
            "optimizer_step": self.state.optimizer_step,
            "tokens_seen": self.state.tokens_seen,
            "validation_loss": round((self.state.last_loss or 0.0) + 0.123, 6),
        }
        self.validation_history.append(metric)
        return metric

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "state": asdict(self.state),
            "records": [asdict(r) for r in self.records],
            "validation_history": self.validation_history,
        }

    def save_checkpoint(self, name: str) -> Path:
        path = self.checkpoint_dir / f"{name}.json"
        path.write_text(json.dumps(self.checkpoint_payload(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        payload = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        self.state = HarnessState(**payload["state"])
        self.records = [MicrobatchRecord(**r) for r in payload["records"]]
        self.validation_history = payload["validation_history"]
        self.random = random.Random(self.state.rng_seed)
        self.peak_memory_bytes = self.state.peak_memory_bytes

    def run(self, stop_after_microbatches: int | None = None) -> None:
        self.snapshot_config()
        while self.state.optimizer_step < self.config.max_optimizer_steps:
            if stop_after_microbatches is not None and self.state.microbatch_count >= stop_after_microbatches:
                break
            self.run_microbatch()
            optimizer_stepped = self.maybe_step_optimizer()
            if optimizer_stepped and self.state.optimizer_step % self.config.validation_interval == 0:
                self.validate()
            if optimizer_stepped and self.state.optimizer_step % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"step_{self.state.optimizer_step:07d}")


def load_harness_config(path: str | Path) -> HarnessConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return HarnessConfig(**payload)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the HZ-0A restart deterministic harness.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--stop-after-microbatches", type=int, default=None)
    args = parser.parse_args()

    cfg = load_harness_config(args.config)
    harness = DeterministicHarness(cfg, args.run_dir)
    if args.resume:
        harness.load_checkpoint(args.resume)
    harness.run(stop_after_microbatches=args.stop_after_microbatches)
    if harness.state.optimizer_step > 0:
        harness.save_checkpoint(f"step_{harness.state.optimizer_step:07d}")
    summary = {
        "optimizer_step": harness.state.optimizer_step,
        "microbatch_count": harness.state.microbatch_count,
        "tokens_seen": harness.state.tokens_seen,
        "effective_batch_tokens": harness.state.effective_batch_tokens,
        "epoch_or_data_pass": harness.state.epoch_or_data_pass,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
