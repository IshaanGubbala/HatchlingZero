"""HZ-0C C6: HZ-0B session-local memory wired through the trigger graph
(scripts/hz0c_c6_conditional_attention_eval.py::conditional_forward_with_memory).

Checked against the ACTUAL frozen HZ-0A checkpoint, not synthetic hidden
states, matching this project's established convention for integration
tests (tests/reference/test_hz0b_b6_real_integration.py). Skips if the
checkpoint isn't present (gitignored under outputs/).
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from reference.hz0b_b8_latent_write import init_latent_write_controller
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences
from scripts.hz0c_c6_conditional_attention_eval import (
    ATTENTION_INDICES, conditional_forward, conditional_forward_with_memory, fixed_matched_trigger,
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists() or not Path(GENERAL_DATA_PATH).exists(),
    reason="frozen HZ-0A checkpoint / real corpus data not present locally (both gitignored)",
)


def _load_tokens(count: int = 4) -> mx.array:
    sequences = load_real_sequences(GENERAL_DATA_PATH, count)
    return mx.array(sequences, dtype=mx.int32)


def test_first_position_read_is_exactly_unaffected_by_memory_regardless_of_write_bias():
    """`latent_write_and_read_step` reads against the PRE-write memory
    state (B1 decision 7's write-visibility convention), and position 0's
    read happens before ANY position has had a chance to write --
    `sequential_latent_write_and_read` starts every sequence from an
    all-zero-confidence bank. `gated_memory_read`'s `confidence_scaled`
    gate is retrieval confidence times the learned gate (its own
    docstring), and retrieval confidence against an all-empty bank is
    exactly 0 by construction (`memory_read`'s weighted sum over
    per-slot confidence, all zero) -- so position 0's memory-wired output
    equals the no-memory output EXACTLY, for any write-gate bias,
    trained or not. This is the write+read analog of B6's own "empty
    memory behaves exactly like no memory" invariant, and does not
    depend on writes being suppressed anywhere (they aren't, at this
    checkpoint's hidden-state scale -- confirmed empirically the write
    gate can saturate near 1.0 even with a -30 bias, since the learned
    projection's raw dot product with a real hidden state dominates a
    constant bias)."""
    model, _ = load_frozen_model()
    tokens = _load_tokens(2)
    trigger = fixed_matched_trigger(*tokens.shape, rate=0.15)
    no_memory_logits = conditional_forward(model, tokens, trigger)
    for bias in (-30.0, -3.0, 0.0):
        latent_params = init_latent_write_controller(model.dim, 32, 32, seed=17, write_gate_bias_init=bias)
        memory_logits, _ = conditional_forward_with_memory(model, tokens, trigger, latent_params)
        mx.eval(no_memory_logits, memory_logits)
        assert bool(mx.array_equal(no_memory_logits[:, 0], memory_logits[:, 0])), f"bias={bias}"


def test_default_write_bias_actually_engages_memory():
    """The project's own established `write_gate_bias_init=-3.0` default
    (`init_latent_write_controller`'s docstring: sigmoid(-3)=0.047, a
    real nonzero per-step write probability, not a disabled one) should
    make the memory-wired forward differ measurably from the no-memory
    forward -- a sanity check that the new wiring is not silently a
    no-op at the convention this project actually uses everywhere else."""
    model, _ = load_frozen_model()
    tokens = _load_tokens(2)
    trigger = fixed_matched_trigger(*tokens.shape, rate=0.15)
    no_memory_logits = conditional_forward(model, tokens, trigger)
    latent_params = init_latent_write_controller(model.dim, 32, 32, seed=17, write_gate_bias_init=-3.0)
    memory_logits, write_gates = conditional_forward_with_memory(model, tokens, trigger, latent_params)
    mx.eval(no_memory_logits, memory_logits, write_gates)
    assert float(mx.mean(write_gates)) > 1e-3
    assert not bool(mx.allclose(no_memory_logits, memory_logits, atol=1e-6))
    assert bool(mx.all(mx.isfinite(memory_logits)))


def test_memory_wired_forward_is_deterministic_given_fixed_seed():
    """Real-inference-time determinism requirement (Hard Constraint:
    "Inference triggering must be deterministic and reproducible") --
    checked here for the full memory-wired graph, not just the trigger
    signal alone, since memory adds its own write/read state that must
    not introduce nondeterminism."""
    model, _ = load_frozen_model()
    tokens = _load_tokens(2)
    trigger = fixed_matched_trigger(*tokens.shape, rate=0.15)
    params_a = init_latent_write_controller(model.dim, 32, 32, seed=17, write_gate_bias_init=-3.0)
    params_b = init_latent_write_controller(model.dim, 32, 32, seed=17, write_gate_bias_init=-3.0)
    logits_a, gates_a = conditional_forward_with_memory(model, tokens, trigger, params_a)
    logits_b, gates_b = conditional_forward_with_memory(model, tokens, trigger, params_b)
    mx.eval(logits_a, logits_b, gates_a, gates_b)
    assert bool(mx.array_equal(logits_a, logits_b))
    assert bool(mx.array_equal(gates_a, gates_b))


def test_attention_indices_still_match_the_frozen_checkpoints_own_schedule():
    """Regression lock on the constant the whole conditional/memory graph
    depends on -- if this ever drifts from the checkpoint's real
    six-layer attention schedule, every C6/C9 result silently becomes
    invalid."""
    assert ATTENTION_INDICES == (4, 9, 14, 19, 24, 29)
