from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.hz0a_gdn2_reference import HZ0ABlock, TinyHZ0AModel, gdn2_scan, init_state  # noqa: E402
from restart.hz0a_pmetal.python.pmetal_reference import (  # noqa: E402
    BlockForwardInputs,
    Gdn2ForwardInputs,
    TinyModelForwardInputs,
    block_forward,
    AdamWState,
    adamw_step,
    gdn2_forward,
    gdn2_backward,
    tiny_model_forward,
)


def test_pmetal_style_forward_matches_numpy_oracle() -> None:
    rng = np.random.default_rng(42)
    q = rng.normal(size=(2, 5, 3, 4)).astype(np.float32)
    k = rng.normal(size=(2, 5, 3, 4)).astype(np.float32)
    v = rng.normal(size=(2, 5, 3, 6)).astype(np.float32)
    decay = rng.normal(size=(2, 5, 3, 4)).astype(np.float32)
    erase = rng.normal(size=(2, 5, 3, 4)).astype(np.float32)
    write = rng.normal(size=(2, 5, 3, 6)).astype(np.float32)
    state = init_state(batch_size=2, num_heads=3, d_v=6, d_k=4)

    expected_out, expected_state = gdn2_scan(q, k, v, decay, erase, write, initial_state=state)
    actual = gdn2_forward(
        Gdn2ForwardInputs(
            q=q,
            k=k,
            v=v,
            decay_logits=decay,
            erase_logits=erase,
            write_logits=write,
            initial_state=state,
        )
    )

    np.testing.assert_allclose(actual.outputs, expected_out, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.final_state, expected_state, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.backward_cache.q, q)
    np.testing.assert_allclose(actual.backward_cache.initial_state, state)


def test_pmetal_style_backward_matches_a3_autodiff_contract() -> None:
    import torch

    torch.manual_seed(12)
    q = torch.randn(1, 4, 2, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 4, 2, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 4, 2, 2, dtype=torch.float64, requires_grad=True)
    decay = torch.randn(1, 4, 2, 3, dtype=torch.float64, requires_grad=True)
    erase = torch.randn(1, 4, 2, 3, dtype=torch.float64, requires_grad=True)
    write = torch.randn(1, 4, 2, 2, dtype=torch.float64, requires_grad=True)
    initial = torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    def torch_gdn2_scan(q_, k_, v_, decay_, erase_, write_, state_):
        outputs = []
        state = state_
        for t in range(q_.shape[1]):
            d = torch.sigmoid(decay_[:, t])
            e = torch.sigmoid(erase_[:, t])
            w = torch.sigmoid(write_[:, t])
            state = d[:, :, None, :] * (1.0 - e[:, :, None, :]) * state
            state = state + w[:, :, :, None] * v_[:, t, :, :, None] * k_[:, t, :, None, :]
            outputs.append(torch.einsum("bhvk,bhk->bhv", state, q_[:, t]))
        return torch.stack(outputs, dim=1), state

    expected_out, expected_state = torch_gdn2_scan(q, k, v, decay, erase, write, initial)
    grad_out = torch.randn_like(expected_out)
    grad_state = torch.randn_like(expected_state)
    (expected_out * grad_out).sum().add((expected_state * grad_state).sum()).backward()
    actual = gdn2_backward(
        grad_out.detach().numpy(),
        grad_state.detach().numpy(),
        gdn2_forward(Gdn2ForwardInputs(q.detach().numpy(), k.detach().numpy(), v.detach().numpy(), decay.detach().numpy(), erase.detach().numpy(), write.detach().numpy(), initial.detach().numpy())).backward_cache,
    )
    for name, tensor in (("q", q), ("k", k), ("v", v), ("decay_logits", decay), ("erase_logits", erase), ("write_logits", write), ("initial_state", initial)):
        np.testing.assert_allclose(actual.gradients[name], tensor.grad.detach().numpy(), rtol=1e-8, atol=1e-8)


def test_pmetal_style_recurrent_block_matches_reference_block() -> None:
    rng = np.random.default_rng(7)
    block = HZ0ABlock.init(
        rng=rng,
        d_model=16,
        num_heads=4,
        d_k=4,
        d_v=4,
        d_ff=32,
        is_attention=False,
    )
    x = rng.normal(size=(2, 5, 16)).astype(np.float32)
    state = init_state(batch_size=2, num_heads=4, d_v=4, d_k=4)
    expected_out, expected_state = block(x, state)
    actual = block_forward(BlockForwardInputs(block=block, x=x, state=state))
    np.testing.assert_allclose(actual.outputs, expected_out, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.final_state, expected_state, rtol=1e-6, atol=1e-6)


def test_pmetal_style_attention_block_matches_reference_block() -> None:
    rng = np.random.default_rng(9)
    block = HZ0ABlock.init(
        rng=rng,
        d_model=12,
        num_heads=3,
        d_k=4,
        d_v=4,
        d_ff=24,
        is_attention=True,
    )
    x = rng.normal(size=(2, 4, 12)).astype(np.float32)
    expected_out, expected_state = block(x, None)
    actual = block_forward(BlockForwardInputs(block=block, x=x, state=None))
    np.testing.assert_allclose(actual.outputs, expected_out, rtol=1e-6, atol=1e-6)
    assert expected_state is None
    assert actual.final_state is None


def test_pmetal_style_tiny_model_loss_matches_reference() -> None:
    model = TinyHZ0AModel.init(
        rng_seed=1234,
        vocab_size=32,
        d_model=16,
        num_layers=3,
        num_heads=4,
        d_k=4,
        d_v=4,
        d_ff=32,
        attention_layer_indices=[1],
    )
    token_ids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64)
    targets = np.array([[2, 3, 4, 5], [3, 2, 1, 0]], dtype=np.int64)

    expected_logits, expected_states = model(token_ids)
    actual = tiny_model_forward(TinyModelForwardInputs(model=model, token_ids=token_ids, targets=targets))

    np.testing.assert_allclose(actual.logits, expected_logits, rtol=1e-6, atol=1e-6)
    for actual_state, expected_state in zip(actual.states, expected_states):
        if expected_state is None:
            assert actual_state is None
        else:
            np.testing.assert_allclose(actual_state, expected_state, rtol=1e-6, atol=1e-6)
    assert actual.loss is not None
    assert np.isfinite(actual.loss)


def test_adamw_step_matches_first_step_reference_update() -> None:
    parameters = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    gradients = np.array([0.25, -0.5, 1.0], dtype=np.float32)

    result = adamw_step(parameters, gradients, learning_rate=1e-3, weight_decay=0.1)
    expected_update = 1e-3 * (np.sign(gradients) + 0.1 * parameters)

    np.testing.assert_allclose(result.parameters, parameters - expected_update, rtol=1e-6, atol=1e-6)
    assert result.state.step == 1
    np.testing.assert_allclose(result.state.first_moment, 0.1 * gradients)
    np.testing.assert_allclose(result.state.second_moment, 0.001 * gradients**2)
    np.testing.assert_allclose(result.update_norm, np.linalg.norm(expected_update), rtol=1e-6, atol=1e-9)
