"""HZ-0D D5: HZ-0C dependency gate (docs/restart/hz0d_d5_dependency_gate_results.md).

Locks in the "trained checkpoint" prerequisite as a real, running check
-- not just a doc claim -- matching this project's established
real-checkpoint-integration-test convention
(tests/reference/test_hz0c_c6_memory_integration.py). Skips if the
checkpoint isn't present locally (gitignored under outputs/).
"""
from __future__ import annotations

import mlx.core as mx
import pytest
import mlx.utils

from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists(),
    reason="frozen HZ-0A checkpoint not present locally (gitignored under outputs/)",
)


def test_frozen_checkpoint_loads_with_real_weights_and_runs_a_finite_forward_pass():
    """The D5 gate's "trained checkpoint" prerequisite, checked directly:
    load the real checkpoint via its own `state.json` array manifest
    (not a synthetic/stub state), run one real forward pass, and confirm
    the output is finite and shaped consistently with the checkpoint's
    OWN embedding weight -- not an assumed vocab size."""
    model, payload = load_frozen_model()
    assert payload["step"] > 0
    assert payload["tokens_seen"] > 0

    vocab_size = model.embedding.weight.shape[0]
    tokens = mx.array([[1, 45, 982, 12, 7, 300, 44, 1023]])
    logits, _next_states = model(tokens)
    mx.eval(logits)

    assert logits.shape == (1, tokens.shape[1], vocab_size)
    assert bool(mx.all(mx.isfinite(logits)))


def test_frozen_checkpoint_has_the_documented_301m_topology():
    """Cross-check against `docs/restart/hz0c_c1_topology.md`'s audited
    parameter count for model 2 (fixed periodic anchors, the frozen
    checkpoint's own architecture): 301,178,112. A real count from the
    real loaded weights, not the doc's own citation trusted blindly."""
    model, _payload = load_frozen_model()
    total_params = sum(value.size for _key, value in mlx.utils.tree_flatten(model.parameters()))
    assert total_params == 301_178_112
