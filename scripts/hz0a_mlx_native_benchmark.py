"""Compare clean MLX recurrent inference with the native Metal recurrence."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    tokens = mx.arange(args.sequence_length).reshape(1, args.sequence_length) % 64
    reference = HZ0AMlxModel(64, 32, 3, 4, 64, ())
    native = HZ0AMlxModel(64, 32, 3, 4, 64, (), native_metal=True)
    native.update(reference.parameters())
    reference_time, reference_logits, reference_states = measure(reference, tokens, args.iterations)
    native_time, native_logits, native_states = measure(native, tokens, args.iterations)
    max_logit_error = float(mx.max(mx.abs(reference_logits - native_logits)))
    max_state_error = max(float(mx.max(mx.abs(a - b))) for a, b in zip(reference_states, native_states))
    print(json.dumps({"sequence_length": args.sequence_length, "iterations": args.iterations, "reference_seconds": reference_time, "native_metal_seconds": native_time, "native_speedup": reference_time / native_time, "max_logit_error": max_logit_error, "max_state_error": max_state_error}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
