import numpy as np

from restart.hz0a_pmetal.python.native_model import NativeTinyHZ0AModel


def test_native_complete_model_forward_backward_and_state_carry():
    model = NativeTinyHZ0AModel(32, 16, 3, 2, 8, 8, 32, [1], seed=9)
    tokens = np.arange(8).reshape(2, 4) % 32
    targets = np.roll(tokens, -1, axis=1)
    loss, states = model.loss_and_backward(tokens, targets)
    assert np.isfinite(loss)
    assert states[0] is not None and states[1] is None and states[2] is not None
    assert all(np.isfinite(parameter.grad).all() for parameter in model.parameters())
    for parameter in model.parameters():
        parameter.zero_grad()
    continued_loss, continued_states = model.loss_and_backward(tokens[:, :1], targets[:, :1], states)
    assert np.isfinite(continued_loss)
    assert continued_states[0].shape == states[0].shape
