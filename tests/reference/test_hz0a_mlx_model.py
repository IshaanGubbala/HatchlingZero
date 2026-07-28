import mlx.core as mx

from reference.hz0a_mlx_model import HZ0AMlxModel


def test_mlx_scaled_model_forward_and_state_carry():
    model = HZ0AMlxModel(32, 16, 3, 2, 32, (1,))
    tokens = mx.arange(10).reshape(1, 10) % 32
    logits, states = model(tokens)
    mx.eval(logits, *[state for state in states if state is not None])
    assert logits.shape == (1, 10, 32)
    assert states[0].shape == (1, 2, 8, 8)
    assert states[1] is None
    assert bool(mx.all(mx.isfinite(logits)))
