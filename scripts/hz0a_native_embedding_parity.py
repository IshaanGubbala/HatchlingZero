"""Emit a machine-readable parity report for native embedding/LM-head training."""

from __future__ import annotations

import hashlib
import json
import resource
import time
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restart.hz0a_pmetal.python.native_layers import NativeEmbedding, NativeTiedLMHead, cross_entropy_backward, cross_entropy_forward


def main() -> None:
    started = time.perf_counter()
    rng = np.random.default_rng(101)
    native = NativeEmbedding("embedding", 16, 8, rng)
    native.weight.data[:] = rng.normal(0.0, 0.02, native.weight.data.shape).astype(np.float32)
    head = NativeTiedLMHead(native)
    ids = np.arange(8).reshape(2, 4) % 16
    targets = np.roll(ids, -1, axis=1)
    logits = head.forward(native.forward(ids))
    loss, cache = cross_entropy_forward(logits, targets)
    grad_hidden = head.backward(cross_entropy_backward(cache))
    native.backward(grad_hidden)
    weight = torch.tensor(native.weight.data, requires_grad=True)
    hidden = weight[torch.tensor(ids)]
    torch_logits = hidden @ weight.T
    torch_loss = torch.nn.functional.cross_entropy(torch_logits.reshape(-1, 16), torch.tensor(targets).reshape(-1))
    torch_loss.backward()
    native_update = 1e-4 * native.weight.grad
    torch_update = 1e-4 * weight.grad.detach().numpy()
    report = {"max_absolute_output_error": float(np.max(np.abs(logits - torch_logits.detach().numpy()))), "max_relative_output_error": float(np.max(np.abs(logits - torch_logits.detach().numpy()) / np.maximum(np.abs(torch_logits.detach().numpy()), 1e-8))), "loss": loss, "loss_difference": abs(loss - float(torch_loss.detach())), "parameter_gradient_errors": {"embedding.weight": float(np.max(np.abs(native.weight.grad - weight.grad.detach().numpy())))}, "update_norm_difference": abs(float(np.linalg.norm(native_update)) - float(np.linalg.norm(torch_update))), "parameter_fingerprint_after_update": hashlib.sha256((native.weight.data - native_update).tobytes()).hexdigest(), "finite": bool(np.isfinite(logits).all() and np.isfinite(native.weight.grad).all()), "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "execution_seconds": time.perf_counter() - started}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
