"""Small explicit native-training primitives used before full PMetal assembly.

These NumPy layers own parameters and manual gradients. Torch remains the
independent oracle in the parity tests; this module intentionally has no
autograd dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class NativeParameter:
    name: str
    data: np.ndarray
    grad: np.ndarray

    @classmethod
    def init(cls, name: str, shape: tuple[int, ...], rng: np.random.Generator, std: float = 0.02) -> "NativeParameter":
        data = rng.normal(0.0, std, size=shape).astype(np.float32)
        return cls(name, data, np.zeros_like(data))

    def zero_grad(self) -> None:
        self.grad.fill(0.0)

    def state_dict(self) -> dict:
        return {"name": self.name, "shape": list(self.data.shape), "data": self.data.tolist(), "grad": self.grad.tolist()}


class NativeLinear:
    def __init__(self, name: str, in_features: int, out_features: int, rng: np.random.Generator, bias: bool = True):
        self.weight = NativeParameter.init(f"{name}.weight", (out_features, in_features), rng)
        self.bias = NativeParameter.init(f"{name}.bias", (out_features,), rng, std=0.0) if bias else None
        self._input = None

    def parameters(self) -> list[NativeParameter]:
        return [self.weight] + ([self.bias] if self.bias is not None else [])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = np.asarray(x, dtype=np.float32)
        output = self._input @ self.weight.data.T
        return output + self.bias.data if self.bias is not None else output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_output = np.asarray(grad_output, dtype=np.float32)
        x2 = self._input.reshape(-1, self._input.shape[-1])
        g2 = grad_output.reshape(-1, grad_output.shape[-1])
        self.weight.grad += (g2.astype(np.float64).T @ x2.astype(np.float64)).astype(np.float32).reshape(self.weight.data.shape)
        if self.bias is not None:
            self.bias.grad += g2.astype(np.float64).sum(axis=0).astype(np.float32)
        return (g2.astype(np.float64) @ self.weight.data.astype(np.float64)).astype(np.float32).reshape(self._input.shape)


class NativeEmbedding:
    def __init__(self, name: str, vocab_size: int, dim: int, rng: np.random.Generator):
        self.weight = NativeParameter.init(f"{name}.weight", (vocab_size, dim), rng)
        self._ids = None

    def parameters(self) -> list[NativeParameter]:
        return [self.weight]

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        self._ids = np.asarray(token_ids, dtype=np.int64)
        return self.weight.data[self._ids]

    def backward(self, grad_output: np.ndarray) -> None:
        np.add.at(self.weight.grad, self._ids, np.asarray(grad_output, dtype=np.float32))


class NativeTiedLMHead:
    def __init__(self, embedding: NativeEmbedding):
        self.embedding = embedding
        self._hidden = None

    def forward(self, hidden: np.ndarray) -> np.ndarray:
        self._hidden = np.asarray(hidden, dtype=np.float32)
        return self._hidden @ self.embedding.weight.data.T

    def backward(self, grad_logits: np.ndarray) -> np.ndarray:
        g2 = np.asarray(grad_logits, dtype=np.float32).reshape(-1, grad_logits.shape[-1])
        h2 = self._hidden.reshape(-1, self._hidden.shape[-1])
        self.embedding.weight.grad += (g2.astype(np.float64).T @ h2.astype(np.float64)).astype(np.float32)
        return (g2.astype(np.float64) @ self.embedding.weight.data.astype(np.float64)).astype(np.float32).reshape(self._hidden.shape)


@dataclass
class CrossEntropyCache:
    probabilities: np.ndarray
    targets: np.ndarray


def cross_entropy_forward(logits: np.ndarray, targets: np.ndarray) -> tuple[float, CrossEntropyCache]:
    values = np.asarray(logits, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=-1, keepdims=True)
    loss = -np.log(np.take_along_axis(probabilities, targets[..., None], axis=-1).clip(1e-30, None)).mean()
    return float(loss), CrossEntropyCache(probabilities, targets)


def cross_entropy_backward(cache: CrossEntropyCache) -> np.ndarray:
    gradient = cache.probabilities.copy()
    flat = gradient.reshape(-1, gradient.shape[-1])
    flat[np.arange(flat.shape[0]), cache.targets.reshape(-1)] -= 1.0
    return (flat / cache.targets.size).reshape(gradient.shape)


def serialize_parameters(parameters: list[NativeParameter], path: str | Path) -> None:
    Path(path).write_text(json.dumps({parameter.name: parameter.state_dict() for parameter in parameters}), encoding="utf-8")
