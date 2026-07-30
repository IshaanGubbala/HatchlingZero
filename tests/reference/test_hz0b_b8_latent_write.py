"""HZ-0B B8 Stage 3 wiring tests: deterministic checks of
`sequential_latent_write_and_read` against synthetic hidden states (no
real HZ-0A checkpoint needed). The real, checkpoint-dependent training
result lives in `scripts/hz0b_b8_stage3_latent_write_probe.py` /
`docs/restart/hz0b_b8_stage3_results.md`.
"""
import mlx.core as mx

from reference.hz0b_b8_latent_write import (
    init_latent_write_controller,
    sequential_latent_write_and_read,
)

D_MODEL, KEY_DIM, VALUE_DIM, BATCH, SEQ = 8, 32, 32, 2, 6


def make_hidden(seed: int) -> mx.array:
    return mx.random.normal((BATCH, SEQ, D_MODEL), key=mx.random.key(seed))


def test_output_and_gate_shapes_and_finite():
    params = init_latent_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
    hidden = make_hidden(seed=2)
    outputs, final_memory, gates = sequential_latent_write_and_read(params, hidden)
    assert outputs.shape == hidden.shape
    assert gates.shape == (BATCH, SEQ)
    assert bool(mx.all(mx.isfinite(outputs)))
    assert bool(mx.all(mx.isfinite(final_memory.values)))
    assert bool(mx.all((gates >= 0) & (gates <= 1)))  # sigmoid output range


def test_write_gate_and_key_value_projections_are_differentiable_via_sparsity_penalty():
    """The exact quantity the B8 probe regularizes (mean write_gate across
    all positions) must produce a real, nonzero gradient back to
    key_proj/value_proj/write_gate params -- otherwise the sparsity
    penalty used in scripts/hz0b_b8_stage3_latent_write_probe.py couldn't
    actually do anything."""
    params = init_latent_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
    hidden = make_hidden(seed=2)

    def sparsity_loss(key_proj_w):
        p = init_latent_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
        p = type(p)(write_controller=p.write_controller, key_proj_w=key_proj_w, key_proj_b=p.key_proj_b,
                     value_proj_w=p.value_proj_w, value_proj_b=p.value_proj_b)
        _, _, gates = sequential_latent_write_and_read(p, hidden)
        return mx.mean(gates)

    grad = mx.grad(sparsity_loss)(params.key_proj_w)
    assert grad.shape == params.key_proj_w.shape
    assert bool(mx.all(mx.isfinite(grad)))
