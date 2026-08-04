"""C6 invariant: HZ-0B memory state is unchanged by chunk boundaries."""
from __future__ import annotations

import json

import mlx.core as mx

from reference.hz0b_b6_hz0a_integration import frozen_hidden_states
from reference.hz0b_b8_latent_write import init_latent_write_controller, latent_write_and_read_step
from reference.hz0b_memory_simulator import MemoryState, reset
from scripts.hz0b_b11_baseline_comparison import load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences


def run_memory(hidden: mx.array, params, *, chunk_size: int | None) -> tuple[mx.array, MemoryState]:
    batch, seq, dim = hidden.shape
    state = reset(batch, 8, 32, 32)
    outputs = []
    ranges = [(0, seq)] if chunk_size is None else [(start, min(start + chunk_size, seq)) for start in range(0, seq, chunk_size)]
    step = 0
    for start, end in ranges:
        for position in range(start, end):
            output, state, _, _ = latent_write_and_read_step(params, hidden[:, position, :], state, step=step, ste=False)
            outputs.append(output)
            state = state
            step += 1
        mx.eval(state.keys, state.values, state.confidence, state.age, state.protection, state.write_count, state.last_write_step, state.write_source)
    return mx.stack(outputs, axis=1), state


def max_state_error(a: MemoryState, b: MemoryState) -> float:
    fields = ("keys", "values", "confidence", "age", "protection", "write_count", "last_write_step", "write_source")
    return max(float(mx.max(mx.abs(getattr(a, field) - getattr(b, field)))) for field in fields)


def main() -> None:
    model, _ = load_frozen_model()
    tokens = mx.array(load_real_sequences(GENERAL_DATA_PATH, 2), dtype=mx.int32)
    hidden, _ = frozen_hidden_states(model, tokens)
    mx.eval(hidden)
    params = init_latent_write_controller(model.dim, 32, 32, seed=17, write_gate_bias_init=-3.0)
    full_output, full_state = run_memory(hidden, params, chunk_size=None)
    chunked_output, chunked_state = run_memory(hidden, params, chunk_size=16)
    mx.eval(full_output, chunked_output)
    report = {
        "sequence_length": int(tokens.shape[1]),
        "chunk_size": 16,
        "max_output_abs_error": float(mx.max(mx.abs(full_output - chunked_output))),
        "max_state_abs_error": max_state_error(full_state, chunked_state),
        "finite": bool(mx.all(mx.isfinite(chunked_output))),
        "exact_within_tolerance": bool(mx.allclose(full_output, chunked_output, atol=1e-6, rtol=1e-6)) and max_state_error(full_state, chunked_state) <= 1e-6,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["exact_within_tolerance"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
