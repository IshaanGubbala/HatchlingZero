"""Emit full tiny native-model versus Torch-oracle parity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0a_torch_model import HZ0AConfig, HZ0AModel
from restart.hz0a_pmetal.python.native_model import NativeTinyHZ0AModel
from restart.hz0a_pmetal.python.training import PmetalOptimizerPath


def max_errors(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    delta = np.abs(left - right)
    return float(delta.max()), float((delta / np.maximum(np.abs(right), 1e-8)).max())


def torch_parameters(model: HZ0AModel) -> list[torch.Tensor]:
    values = [model.embedding.weight]
    for block in model.blocks:
        values.append(block.norm1.weight)
        values.extend(block.mixer.parameters())
        values.append(block.norm2.weight)
        values.extend(block.mlp.parameters())
    values.append(model.final_norm.weight)
    return values


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes()).hexdigest()


def run() -> dict:
    started = time.perf_counter()
    torch.manual_seed(99)
    config = HZ0AConfig(32, 16, 3, 2, 8, 8, 32, (1,))
    oracle = HZ0AModel(config)
    native = NativeTinyHZ0AModel(32, 16, 3, 2, 8, 8, 32, [1], seed=99)
    native_parameters = native.parameters()
    oracle_parameters = torch_parameters(oracle)
    for native_parameter, oracle_parameter in zip(native_parameters, oracle_parameters):
        native_parameter.data[...] = oracle_parameter.detach().numpy()
    tokens = np.arange(8).reshape(2, 4) % 32
    targets = np.roll(tokens, -1, axis=1)
    native_loss, _ = native.loss_and_backward(tokens, targets)
    native_logits, _ = native.forward(tokens)
    torch_tokens, torch_targets = torch.tensor(tokens), torch.tensor(targets)
    oracle_logits, _ = oracle(torch_tokens)
    oracle_loss = torch.nn.functional.cross_entropy(oracle_logits.reshape(-1, 32), torch_targets.reshape(-1))
    oracle_loss.backward()
    native_flat_grad = np.concatenate([p.grad.reshape(-1) for p in native_parameters]).astype(np.float64)
    oracle_flat_grad = np.concatenate([p.grad.detach().numpy().reshape(-1) for p in oracle_parameters]).astype(np.float64)
    grad_abs, grad_rel = max_errors(native_flat_grad, oracle_flat_grad)
    before = native.flat_parameters()
    native_optimizer = PmetalOptimizerPath(before, total_steps=1, learning_rate=1e-4, weight_decay=0.01)
    native_update = native_optimizer.add_microbatch(native_flat_grad, tokens=tokens.size)
    native.load_flat_parameters(native_optimizer.state.parameters)
    torch_optimizer = torch.optim.AdamW(oracle.parameters(), lr=1e-4, weight_decay=0.01)
    torch_optimizer.step()
    oracle_after = np.concatenate([p.detach().numpy().reshape(-1) for p in oracle_parameters]).astype(np.float64)
    native_after = native.flat_parameters()
    update_abs, update_rel = max_errors(native_after, oracle_after)
    oracle_delta = before - oracle_after
    native_delta = before - native_after
    return {
        "max_absolute_output_error": max_errors(native_logits, oracle_logits.detach().numpy())[0],
        "max_relative_output_error": max_errors(native_logits, oracle_logits.detach().numpy())[1],
        "loss": native_loss,
        "loss_difference": abs(native_loss - float(oracle_loss.detach())),
        "max_absolute_gradient_error": grad_abs,
        "max_relative_gradient_error": grad_rel,
        "per_parameter_gradient_errors": {p.name: max_errors(p.grad, q.grad.detach().numpy())[0] for p, q in zip(native_parameters, oracle_parameters)},
        "update_norm_difference": abs(float(np.linalg.norm(native_delta)) - float(np.linalg.norm(oracle_delta))),
        "parameter_fingerprint_after_update": fingerprint(native_after),
        "torch_parameter_fingerprint_after_update": fingerprint(oracle_after),
        "parameter_update_max_absolute_error": update_abs,
        "parameter_update_max_relative_error": update_rel,
        "optimizer_metrics": native_update,
        "finite": bool(np.isfinite(native_logits).all() and np.isfinite(native_flat_grad).all() and np.isfinite(native_after).all()),
        "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "execution_seconds": time.perf_counter() - started,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))
