"""Manual-backward normalization, activation, residual, and dense MLP blocks."""

from __future__ import annotations

import numpy as np

from restart.hz0a_pmetal.python.native_layers import NativeLinear, NativeParameter


class NativeRMSNorm:
    def __init__(self, name: str, dim: int):
        self.weight = NativeParameter(name + ".weight", np.ones(dim, dtype=np.float32), np.zeros(dim, dtype=np.float32))
        self._x = self._inv_rms = None

    def parameters(self):
        return [self.weight]

    def forward(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        self._x = np.asarray(x, dtype=np.float32)
        self._inv_rms = 1.0 / np.sqrt(np.mean(self._x * self._x, axis=-1, keepdims=True) + eps)
        return self._x * self._inv_rms * self.weight.data

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        g = np.asarray(grad_output, dtype=np.float32)
        normalized = self._x * self._inv_rms
        self.weight.grad += np.sum(g * normalized, axis=tuple(range(g.ndim - 1)))
        dot = np.sum(g * self.weight.data * self._x, axis=-1, keepdims=True)
        dim = self._x.shape[-1]
        return self.weight.data * self._inv_rms * g - self.weight.data * self._x * (self._inv_rms ** 3) * dot / dim


class NativeSiLU:
    def __init__(self):
        self._x = None

    def forward(self, x):
        self._x = np.asarray(x, dtype=np.float32)
        sigmoid = 1.0 / (1.0 + np.exp(-self._x))
        return self._x * sigmoid

    def backward(self, grad_output):
        sigmoid = 1.0 / (1.0 + np.exp(-self._x))
        return np.asarray(grad_output, dtype=np.float32) * sigmoid * (1.0 + self._x * (1.0 - sigmoid))


def residual_forward(x, update):
    return np.asarray(x, dtype=np.float32) + np.asarray(update, dtype=np.float32)


def residual_backward(grad_output):
    return grad_output, grad_output


class NativeSwiGLU:
    def __init__(self, name: str, dim: int, d_ff: int, rng: np.random.Generator):
        self.gate = NativeLinear(name + ".gate", dim, d_ff, rng)
        self.up = NativeLinear(name + ".up", dim, d_ff, rng)
        self.down = NativeLinear(name + ".down", d_ff, dim, rng)
        self.activation = NativeSiLU()
        self._gate = self._up = None

    def parameters(self):
        return self.gate.parameters() + self.up.parameters() + self.down.parameters()

    def forward(self, x):
        self._gate = self.gate.forward(x)
        self._up = self.up.forward(x)
        return self.down.forward(self.activation.forward(self._gate) * self._up)

    def backward(self, grad_output):
        grad_product = self.down.backward(grad_output)
        grad_gate = self.activation.backward(grad_product * self._up)
        grad_up = grad_product * self.activation.forward(self._gate)
        return self.gate.backward(grad_gate) + self.up.backward(grad_up)
