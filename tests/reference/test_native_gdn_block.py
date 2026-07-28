import numpy as np

from restart.hz0a_pmetal.python.native_gdn_block import NativeGDN2Block


def test_native_gdn_block_forward_backward_and_state_carry_are_finite():
    rng = np.random.default_rng(33)
    block = NativeGDN2Block("gdn", 8, 2, 4, 4, rng)
    x = rng.normal(size=(2, 5, 8)).astype(np.float32)
    output, state = block.forward(x)
    grad_input, grad_initial = block.backward(np.ones_like(output))
    assert output.shape == x.shape
    assert state.shape == (2, 2, 4, 4)
    assert grad_input.shape == x.shape
    assert grad_initial.shape == state.shape
    assert np.isfinite(output).all() and np.isfinite(grad_input).all()
    continued, continued_state = block.forward(x[:, :1], state)
    assert continued.shape == (2, 1, 8)
    assert continued_state.shape == state.shape
