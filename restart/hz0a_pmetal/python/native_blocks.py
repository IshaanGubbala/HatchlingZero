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


class NativeTop1MoE:
    """Manual-backward top-1 MoE FFN with deterministic capacity routing."""

    def __init__(self, name, dim, num_experts, expert_d_ff, capacity_factor, rng, fallback_d_ff=None):
        if num_experts <= 0 or expert_d_ff <= 0 or capacity_factor <= 0:
            raise ValueError("MoE dimensions and capacity factor must be positive")
        self.num_experts = num_experts
        self.capacity_factor = float(capacity_factor)
        self.router = NativeLinear(name + ".router", dim, num_experts, rng)
        self.experts = [NativeSwiGLU(f"{name}.experts.{i}", dim, expert_d_ff, rng) for i in range(num_experts)]
        fallback_d_ff = dim if fallback_d_ff is None else int(fallback_d_ff)
        if fallback_d_ff <= 0:
            raise ValueError("fallback hidden width must be positive")
        self.fallback = NativeSwiGLU(name + ".fallback", dim, fallback_d_ff, rng)
        self._cache = None

    def parameters(self):
        parameters = self.router.parameters()
        for expert in self.experts:
            parameters.extend(expert.parameters())
        parameters.extend(self.fallback.parameters())
        return parameters

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if not np.isfinite(x).all():
            raise FloatingPointError("MoE input contains non-finite values")
        flat = x.reshape(-1, x.shape[-1])
        logits = self.router.forward(flat)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("MoE router probabilities contain non-finite values")
        chosen = np.argmax(logits, axis=-1)
        capacity = max(1, int(np.ceil(self.capacity_factor * flat.shape[0] / self.num_experts)))
        ranks = np.zeros(self.num_experts, dtype=np.int64)
        overflow = np.zeros(flat.shape[0], dtype=bool)
        for token, expert in enumerate(chosen):
            overflow[token] = ranks[expert] >= capacity
            ranks[expert] += 1
        gate = probabilities[np.arange(flat.shape[0]), chosen]
        expert_outputs = [expert.forward(flat) for expert in self.experts]
        fallback_output = self.fallback.forward(flat)
        output = fallback_output * overflow[:, None]
        for expert in range(self.num_experts):
            selected = (chosen == expert) & ~overflow
            output += expert_outputs[expert] * (selected * gate)[:, None]
        if not np.isfinite(output).all():
            raise FloatingPointError("MoE output contains non-finite values")
        self._cache = (x, flat, probabilities, chosen, overflow, gate, expert_outputs, fallback_output)
        return output.reshape(x.shape)

    def backward(self, grad_output):
        x, flat, probabilities, chosen, overflow, gate, expert_outputs, fallback_output = self._cache
        grad = np.asarray(grad_output, dtype=np.float32).reshape(flat.shape)
        if not np.isfinite(grad).all():
            raise FloatingPointError("MoE gradient contains non-finite values")
        grad_input = np.zeros_like(flat)
        grad_logits = np.zeros_like(probabilities)
        for expert in range(self.num_experts):
            selected = (chosen == expert) & ~overflow
            scale = (selected * gate)[:, None]
            expert_grad = self.experts[expert].backward(grad * scale)
            # The expert backward already includes the routed output scale.
            grad_input += expert_grad
            grad_gate = np.sum(grad * expert_outputs[expert], axis=-1) * selected
            one_hot = (np.arange(self.num_experts) == expert).astype(np.float32)
            grad_logits += (grad_gate * gate)[:, None] * (one_hot - probabilities)
        fallback_grad = self.fallback.backward(grad * overflow[:, None])
        grad_input += fallback_grad * overflow[:, None]
        grad_input += self.router.backward(grad_logits)
        if not np.isfinite(grad_input).all() or not np.isfinite(grad_logits).all():
            raise FloatingPointError("MoE backward produced non-finite gradients")
        return grad_input.reshape(x.shape)
