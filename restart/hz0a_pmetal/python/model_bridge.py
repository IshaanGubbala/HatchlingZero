"""Full-topology gradient bridge from the reference model into PMetal AdamW arrays.

This is a correctness bridge: forward/backward are Torch autograd, while the
parameter update, accumulation, clipping, scheduler, and checkpoint state use
the PMetal optimizer protocol. Native PMetal tensor execution remains a
separate backend gate.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from reference.hz0a_torch_model import HZ0AConfig, HZ0AModel
from restart.hz0a_pmetal.python.training import PmetalOptimizerPath


class PmetalModelBridge:
    def __init__(self, config_path: str | Path, *, seed: int = 0, accumulation_steps: int = 1):
        torch.manual_seed(seed)
        self.config = HZ0AConfig.from_json(config_path)
        self.model = HZ0AModel(self.config).float()
        self.names = [name for name, _ in self.model.named_parameters()]
        self.shapes = [tuple(parameter.shape) for parameter in self.model.parameters()]
        flat = np.concatenate([parameter.detach().numpy().reshape(-1) for parameter in self.model.parameters()]).astype(np.float64)
        self.optimizer = PmetalOptimizerPath(flat, accumulation_steps=accumulation_steps, total_steps=200)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.parameters().tobytes()).hexdigest()

    def parameters(self) -> np.ndarray:
        return np.concatenate([parameter.detach().numpy().reshape(-1) for parameter in self.model.parameters()]).astype(np.float64)

    def _load_parameters(self, values: np.ndarray) -> None:
        offset = 0
        with torch.no_grad():
            for parameter, shape in zip(self.model.parameters(), self.shapes):
                count = int(np.prod(shape))
                parameter.copy_(torch.from_numpy(values[offset:offset + count].reshape(shape)).float())
                offset += count

    def loss_and_gradients(self, token_ids: np.ndarray, targets: np.ndarray, states=None) -> tuple[float, np.ndarray, list[torch.Tensor | None]]:
        tokens = torch.from_numpy(token_ids.astype(np.int64))
        target_tensor = torch.from_numpy(targets.astype(np.int64))
        logits, next_states = self.model(tokens, states)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_tensor.reshape(-1))
        gradients = torch.autograd.grad(loss, tuple(self.model.parameters()))
        flat = np.concatenate([gradient.detach().numpy().reshape(-1) for gradient in gradients]).astype(np.float64)
        detached_states = [state.detach() if state is not None else None for state in next_states]
        return float(loss.detach()), flat, detached_states

    def train_microbatch(self, token_ids: np.ndarray, targets: np.ndarray, *, tokens_seen: int | None = None, states=None) -> tuple[float, dict | None, list[torch.Tensor | None]]:
        loss, gradients, next_states = self.loss_and_gradients(token_ids, targets, states)
        metric = self.optimizer.add_microbatch(gradients, tokens=int(tokens_seen or token_ids.size))
        if metric is not None:
            self._load_parameters(self.optimizer.state.parameters)
        return loss, metric, next_states
