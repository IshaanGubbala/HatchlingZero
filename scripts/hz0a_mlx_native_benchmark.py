"""Compare clean MLX HZ-0A inference with native Metal recurrence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from reference.hz0a_mlx_model import HZ0AMlxModel


def measure(model: HZ0AMlxModel, tokens, iterations: int) -> tuple[float, object, list]:
    start = time.perf_counter()
    logits = states = None
    for _ in range(iterations):
        logits, states = model(tokens)
        mx.eval(logits, *states)
    return (time.perf_counter() - start) / iterations, logits, states


def measure_decode(model: HZ0AMlxModel, tokens, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        states = None
        for index in range(tokens.shape[1]):
            logits, states = model(tokens[:, index:index + 1], states)
            mx.eval(logits, *[item for state in states for item in (state if isinstance(state, tuple) else (state,)) if item is not None])
    return (time.perf_counter() - start) / iterations


def state_error(reference_states, native_states) -> float:
    errors = []
    for reference, native in zip(reference_states, native_states):
        if reference is None:
            continue
        if isinstance(reference, tuple):
            errors.extend(float(mx.max(mx.abs(a - b))) for a, b in zip(reference, native))
        else:
            errors.append(float(mx.max(mx.abs(reference - native))))
    return max(errors, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--attention-layer", type=int, action="append", default=[])
    args = parser.parse_args()
    tokens = mx.arange(args.sequence_length).reshape(1, args.sequence_length) % 64
    attention = tuple(args.attention_layer)
    reference = HZ0AMlxModel(64, 32, 3, 4, 64, attention)
    native = HZ0AMlxModel(64, 32, 3, 4, 64, attention, native_metal=True)
    native.update(reference.parameters())
    reference_time, reference_logits, reference_states = measure(reference, tokens, args.iterations)
    native_time, native_logits, native_states = measure(native, tokens, args.iterations)
    max_logit_error = float(mx.max(mx.abs(reference_logits - native_logits)))
    max_state_error = state_error(reference_states, native_states)
    reference_decode = measure_decode(reference, tokens, args.iterations)
    native_decode = measure_decode(native, tokens, args.iterations)
    print(json.dumps({"sequence_length": args.sequence_length, "iterations": args.iterations, "attention_layers": attention, "reference_seconds": reference_time, "native_metal_seconds": native_time, "native_speedup": reference_time / native_time, "reference_decode_seconds": reference_decode, "native_decode_seconds": native_decode, "native_decode_speedup": reference_decode / native_decode, "max_logit_error": max_logit_error, "max_state_error": max_state_error}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
