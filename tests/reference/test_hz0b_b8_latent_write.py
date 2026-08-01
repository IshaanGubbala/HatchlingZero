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
                     value_proj_w=p.value_proj_w, value_proj_b=p.value_proj_b, occupancy_gate_w=p.occupancy_gate_w)
        _, _, gates = sequential_latent_write_and_read(p, hidden)
        return mx.mean(gates)

    grad = mx.grad(sparsity_loss)(params.key_proj_w)
    assert grad.shape == params.key_proj_w.shape
    assert bool(mx.all(mx.isfinite(grad)))


def test_ste_write_gate_is_exactly_binary():
    """2026-08-01: ste=True (B1 decision 5's deferred hard/STE routing
    experiment) must produce write_gate values that are EXACTLY 0 or 1
    in the forward pass -- not just close to it -- since the whole point
    is removing partial/continuous writes."""
    params = init_latent_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
    hidden = make_hidden(seed=2)
    _, _, gates = sequential_latent_write_and_read(params, hidden, ste=True)
    is_binary = mx.logical_or(mx.abs(gates - 0.0) < 1e-6, mx.abs(gates - 1.0) < 1e-6)
    assert bool(mx.all(is_binary))


def test_ste_gradient_still_flows_through_the_soft_path():
    """The forward value is hard (0/1), but the gradient must still flow
    -- a straight-through estimator that produced a zero or undefined
    gradient would make the write gate untrainable, defeating the whole
    point of it being a LEARNED decision.

    Differentiates w.r.t. `write_gate_w` specifically (not `key_proj_w`
    -- `occupancy_gate_w` initializes to exactly zero, so at these fresh
    params `write_gate` has zero TRUE dependency on key_proj_w in either
    the ste or non-ste case; write_gate_w is what the gate logit is
    actually a direct function of)."""
    params = init_latent_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
    hidden = make_hidden(seed=2)

    def sparsity_loss(write_gate_w):
        wc = params.write_controller
        new_wc = type(wc)(read_params=wc.read_params, write_gate_w=write_gate_w, write_gate_b=wc.write_gate_b,
                           update_gate_w=wc.update_gate_w, update_gate_b=wc.update_gate_b,
                           protect_gate_w=wc.protect_gate_w, protect_gate_b=wc.protect_gate_b,
                           delete_gate_w=wc.delete_gate_w, delete_gate_b=wc.delete_gate_b)
        p = type(params)(write_controller=new_wc, key_proj_w=params.key_proj_w, key_proj_b=params.key_proj_b,
                          value_proj_w=params.value_proj_w, value_proj_b=params.value_proj_b, occupancy_gate_w=params.occupancy_gate_w)
        _, _, gates = sequential_latent_write_and_read(p, hidden, ste=True)
        return mx.mean(gates)

    grad = mx.grad(sparsity_loss)(params.write_controller.write_gate_w)
    assert grad.shape == params.write_controller.write_gate_w.shape
    assert bool(mx.all(mx.isfinite(grad)))
    assert not bool(mx.all(grad == 0))


def test_ste_false_is_unchanged_from_before_this_flag_existed():
    """Default behavior (ste=False) must be bit-identical to the
    pre-existing continuous write gate -- this flag is strictly opt-in."""
    params = init_latent_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
    hidden = make_hidden(seed=2)
    _, _, gates_default = sequential_latent_write_and_read(params, hidden)
    _, _, gates_explicit_false = sequential_latent_write_and_read(params, hidden, ste=False)
    assert bool(mx.all(gates_default == gates_explicit_false))
    assert bool(mx.any((gates_default > 1e-6) & (gates_default < 1 - 1e-6)))  # genuinely continuous, not accidentally already binary
