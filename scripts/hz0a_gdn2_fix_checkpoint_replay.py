"""Exact checkpoint/resume replay for the native corrected GDN-2 path."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.hz0a_mlx_model import HZ0AMlxModel


def fingerprint(model):
    values = [np.asarray(value).tobytes() for _, value in tree_flatten(model.parameters())]
    return hashlib.sha256(b"".join(values)).hexdigest()


def canonical_fingerprint(model, decimals=6):
    values = [np.round(np.asarray(value), decimals=decimals).astype(np.float32).tobytes() for _, value in tree_flatten(model.parameters())]
    return hashlib.sha256(b"".join(values)).hexdigest()


def parameter_error(left, right):
    left_values = [np.asarray(value) for _, value in tree_flatten(left.parameters())]
    right_values = [np.asarray(value) for _, value in tree_flatten(right.parameters())]
    return max(float(np.max(np.abs(a - b))) for a, b in zip(left_values, right_values))


def snapshot(value):
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot(item) for item in value)
    return np.asarray(value)


def restore(value):
    if isinstance(value, dict):
        return {key: restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [restore(item) for item in value]
    if isinstance(value, tuple):
        return tuple(restore(item) for item in value)
    return mx.array(value)


def save_checkpoint(path, model, optimizer, step, states):
    with path.open("wb") as handle:
        pickle.dump({
            "step": step,
            "parameters": snapshot(model.parameters()),
            "optimizer": snapshot(optimizer.state),
            "states": snapshot(states),
        }, handle)


def load_checkpoint(path, model, optimizer):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    model.update(restore(payload["parameters"]))
    optimizer._state = restore(payload["optimizer"])
    return payload["step"], restore(payload["states"])


def run(steps: int, checkpoint: Path, interrupt: int) -> dict:
    mx.random.seed(41)
    batches = [mx.random.randint(0, 64, (1, 16), key=mx.random.key(1000 + i)) for i in range(steps)]

    def loss_fn(model, tokens):
        logits, _ = model(tokens)
        return mx.mean(nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:]))

    def train_step(model, optimizer, tokens, states=None):
        def stateful_loss(m):
            logits, next_states = m(tokens, states)
            return mx.mean(nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:])), next_states
        # The tiny replay has no recurrent state across batches; state carry is
        # checked separately by the model tests and would confound resume.
        value_grad = nn.value_and_grad(model, lambda m, t: loss_fn(m, t))
        loss, grads = value_grad(model, tokens)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)
        return float(loss)

    full = HZ0AMlxModel(64, 32, 3, 4, 64, (), native_metal=True, mixer="gdn2_fix")
    full_opt = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
    full_losses = [train_step(full, full_opt, batch) for batch in batches]

    mx.random.seed(41)
    resumed = HZ0AMlxModel(64, 32, 3, 4, 64, (), native_metal=True, mixer="gdn2_fix")
    resumed_opt = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
    for index, batch in enumerate(batches[:interrupt], 1):
        train_step(resumed, resumed_opt, batch)
        if index == interrupt:
            mx.eval(resumed.parameters(), resumed_opt.state)
            save_checkpoint(checkpoint, resumed, resumed_opt, index, [])
    saved_step, _ = load_checkpoint(checkpoint, resumed, resumed_opt)
    for batch in batches[saved_step:]:
        train_step(resumed, resumed_opt, batch)
    return {
        "steps": steps,
        "interrupt_step": interrupt,
        "saved_step": saved_step,
        "uninterrupted_fingerprint": fingerprint(full),
        "resumed_fingerprint": fingerprint(resumed),
        "exact_resume": fingerprint(full) == fingerprint(resumed),
        "max_parameter_error": parameter_error(full, resumed),
        "canonical_resume": canonical_fingerprint(full) == canonical_fingerprint(resumed),
        "resume_within_1e-6": parameter_error(full, resumed) <= 1e-6,
        "loss_first": full_losses[0],
        "loss_last": full_losses[-1],
        "finite": all(np.isfinite(full_losses)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--interrupt", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path, default=Path("/tmp/hz0a_gdn2_fix_checkpoint.pkl"))
    args = parser.parse_args()
    started = time.perf_counter()
    report = run(args.steps, args.checkpoint, args.interrupt)
    report["execution_seconds"] = time.perf_counter() - started
    print(json.dumps(report, indent=2, sort_keys=True))
