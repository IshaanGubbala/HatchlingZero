"""Regression test for the confidence-scaling fix
(`reference/hz0b_readonly_integration.py`'s `gated_memory_read`,
`confidence_scaled=True`): a TRAINED (nonzero bias) read path must still
produce EXACT output-equals-hidden behavior against a truly empty memory,
not just an untrained one -- this is what B6/B7's real-integration
write-ups found broken with `confidence_scaled=False` (the default,
unchanged for backward compatibility) once `gate_b`/`value_to_hidden_b`
move away from zero during training. Uses synthetic hidden states and a
hand-perturbed (simulated "trained") params object -- no real HZ-0A
checkpoint needed, fast and deterministic.
"""
import mlx.core as mx

from reference.hz0b_memory_simulator import reset as memory_reset
from reference.hz0b_memory_simulator import write as memory_write
from reference.hz0b_readonly_integration import gated_memory_read, init_readonly_integration

D_MODEL, KEY_DIM, VALUE_DIM, NUM_SLOTS, BATCH = 16, 8, 8, 8, 2


def make_hidden(seed: int) -> mx.array:
    return mx.random.normal((BATCH, D_MODEL), key=mx.random.key(seed))


def simulate_trained_bias(params):
    """A real, non-init bias -- the exact scenario that broke the
    unscaled gate once training moved gate_b/value_to_hidden_b away from
    zero (see docs/restart/hz0b_b7_real_integration_results.md section 2)."""
    from dataclasses import replace
    return replace(params, gate_b=mx.full(params.gate_b.shape, 2.5), value_to_hidden_b=mx.full(params.value_to_hidden_b.shape, 3.7))


def test_unscaled_gate_leaks_on_empty_memory_once_biased_reproducing_the_bug():
    """Confirms the bug this fix addresses is real and reproducible in
    isolation, not just something observed once against the real
    checkpoint."""
    params = simulate_trained_bias(init_readonly_integration(D_MODEL, KEY_DIM, VALUE_DIM, seed=1))
    hidden = make_hidden(seed=2)
    memory = memory_reset(batch_size=BATCH, num_slots=NUM_SLOTS, key_dim=KEY_DIM, value_dim=VALUE_DIM)
    output, _ = gated_memory_read(params, hidden, memory, confidence_scaled=False)
    assert not bool(mx.array_equal(output, hidden))


def test_confidence_scaled_gate_is_exact_on_empty_memory_even_when_biased():
    params = simulate_trained_bias(init_readonly_integration(D_MODEL, KEY_DIM, VALUE_DIM, seed=1))
    hidden = make_hidden(seed=2)
    memory = memory_reset(batch_size=BATCH, num_slots=NUM_SLOTS, key_dim=KEY_DIM, value_dim=VALUE_DIM)
    output, _ = gated_memory_read(params, hidden, memory, confidence_scaled=True)
    assert bool(mx.array_equal(output, hidden))


def test_confidence_scaled_gate_still_reads_real_content_when_memory_is_populated():
    """The fix must not neuter real reads -- a populated, matching slot
    should still contribute (scaled by its own confidence, not zeroed)."""
    params = simulate_trained_bias(init_readonly_integration(D_MODEL, KEY_DIM, VALUE_DIM, seed=1))
    hidden = make_hidden(seed=2)
    memory = memory_reset(batch_size=BATCH, num_slots=NUM_SLOTS, key_dim=KEY_DIM, value_dim=VALUE_DIM)
    key = mx.ones((BATCH, KEY_DIM))
    value = mx.ones((BATCH, VALUE_DIM)) * 5.0
    written, _, _ = memory_write(memory, key, value, mx.array([1.0] * BATCH), step=0, slot_idx=mx.zeros((BATCH,), dtype=mx.int32))
    output, _ = gated_memory_read(params, hidden, written, confidence_scaled=True)
    assert not bool(mx.array_equal(output, hidden))
    assert bool(mx.all(mx.isfinite(output)))


def test_confidence_scaled_gate_is_differentiable():
    params = simulate_trained_bias(init_readonly_integration(D_MODEL, KEY_DIM, VALUE_DIM, seed=1))
    hidden = make_hidden(seed=2)
    memory = memory_reset(batch_size=BATCH, num_slots=NUM_SLOTS, key_dim=KEY_DIM, value_dim=VALUE_DIM)
    key = mx.ones((BATCH, KEY_DIM))
    value = mx.ones((BATCH, VALUE_DIM)) * 5.0
    written, _, _ = memory_write(memory, key, value, mx.array([1.0] * BATCH), step=0, slot_idx=mx.zeros((BATCH,), dtype=mx.int32))

    def loss_fn(query_w):
        p = init_readonly_integration(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
        from dataclasses import replace
        p = replace(p, query_w=query_w)
        output, _ = gated_memory_read(p, hidden, written, confidence_scaled=True)
        return mx.sum(output)

    grad = mx.grad(loss_fn)(params.query_w)
    assert grad.shape == params.query_w.shape
    assert bool(mx.all(mx.isfinite(grad)))
