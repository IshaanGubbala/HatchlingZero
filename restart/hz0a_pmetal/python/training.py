"""Deterministic PMetal-side optimizer protocol for parameter-array replays."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from restart.hz0a_pmetal.python.pmetal_reference import AdamWState, adamw_step


@dataclass
class PmetalTrainingState:
    parameters: np.ndarray
    optimizer: AdamWState | None
    microbatch_count: int = 0
    optimizer_step: int = 0
    tokens_seen: int = 0
    accumulated_gradient: np.ndarray | None = None


class PmetalOptimizerPath:
    def __init__(self, parameters: np.ndarray, *, accumulation_steps: int = 1, max_grad_norm: float = 1.0, learning_rate: float = 1e-4, total_steps: int = 100, weight_decay: float = 0.01):
        if accumulation_steps <= 0 or total_steps <= 0 or max_grad_norm <= 0:
            raise ValueError("accumulation_steps, total_steps, and max_grad_norm must be positive")
        self.state = PmetalTrainingState(np.asarray(parameters).copy(), None, accumulated_gradient=np.zeros_like(parameters, dtype=np.float64))
        self.accumulation_steps, self.max_grad_norm, self.base_learning_rate = accumulation_steps, max_grad_norm, learning_rate
        self.total_steps, self.weight_decay = total_steps, weight_decay

    def add_microbatch(self, gradient: np.ndarray, *, tokens: int) -> dict | None:
        gradient = np.asarray(gradient, dtype=np.float64)
        if gradient.shape != self.state.parameters.shape or not np.isfinite(gradient).all():
            raise FloatingPointError("PMetal refuses non-finite or mismatched gradients")
        self.state.accumulated_gradient += gradient
        self.state.microbatch_count += 1
        self.state.tokens_seen += int(tokens)
        if self.state.microbatch_count % self.accumulation_steps:
            return None
        gradient = self.state.accumulated_gradient / self.accumulation_steps
        unclipped_norm = float(np.linalg.norm(gradient))
        scale = min(1.0, self.max_grad_norm / max(unclipped_norm, 1e-30))
        clipped = gradient * scale
        clipped_norm = float(np.linalg.norm(clipped))
        progress = min(1.0, self.state.optimizer_step / self.total_steps)
        learning_rate = self.base_learning_rate * 0.5 * (1.0 + np.cos(np.pi * progress))
        result = adamw_step(self.state.parameters, clipped, self.state.optimizer, learning_rate=learning_rate, weight_decay=self.weight_decay)
        self.state.parameters, self.state.optimizer = result.parameters, result.state
        self.state.optimizer_step += 1
        self.state.accumulated_gradient.fill(0.0)
        return {"optimizer_step": self.state.optimizer_step, "tokens_seen": self.state.tokens_seen, "learning_rate": learning_rate, "unclipped_gradient_norm": unclipped_norm, "clipped_gradient_norm": clipped_norm, "update_norm": result.update_norm}

    def fingerprint(self) -> str:
        return hashlib.sha256(self.state.parameters.tobytes()).hexdigest()

    def checkpoint(self, path: str | Path) -> None:
        payload = {"parameters": self.state.parameters.tolist(), "optimizer": None if self.state.optimizer is None else {"step": self.state.optimizer.step, "first_moment": self.state.optimizer.first_moment.tolist(), "second_moment": self.state.optimizer.second_moment.tolist()}, "microbatch_count": self.state.microbatch_count, "optimizer_step": self.state.optimizer_step, "tokens_seen": self.state.tokens_seen, "accumulated_gradient": self.state.accumulated_gradient.tolist(), "accumulation_steps": self.accumulation_steps, "max_grad_norm": self.max_grad_norm, "base_learning_rate": self.base_learning_rate, "total_steps": self.total_steps, "weight_decay": self.weight_decay}
        target = Path(path); temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def restore(cls, path: str | Path) -> "PmetalOptimizerPath":
        return cls.restore_payload(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def restore_payload(cls, payload: dict) -> "PmetalOptimizerPath":
        runner = cls(np.asarray(payload["parameters"], dtype=np.float64), accumulation_steps=payload["accumulation_steps"], max_grad_norm=payload["max_grad_norm"], learning_rate=payload["base_learning_rate"], total_steps=payload["total_steps"], weight_decay=payload["weight_decay"])
        runner.state.microbatch_count, runner.state.optimizer_step, runner.state.tokens_seen = payload["microbatch_count"], payload["optimizer_step"], payload["tokens_seen"]
        runner.state.accumulated_gradient = np.asarray(payload["accumulated_gradient"], dtype=np.float64)
        if payload["optimizer"] is not None:
            item = payload["optimizer"]
            runner.state.optimizer = AdamWState(item["step"], np.asarray(item["first_moment"], dtype=np.float64), np.asarray(item["second_moment"], dtype=np.float64))
        return runner
