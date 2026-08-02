import numpy as np
import torch

from reference.hz0a_gdn2_fix_reference import gdn2_fix_chunk_scan, gdn2_fix_scan, gdn2_fix_step
from reference.hz0a_gdn2_fix_torch import gdn2_fix_scan as torch_scan


def test_zero_erase_is_additive_write_and_decay():
    state = np.ones((1, 1, 2, 2), dtype=np.float32)
    q = np.ones((1, 1, 2), dtype=np.float32)
    k = np.array([[[1.0, 0.0]]], dtype=np.float32)
    v = np.array([[[2.0, 3.0]]], dtype=np.float32)
    alpha = np.ones_like(k)
    erase = np.zeros_like(k)
    write = np.ones_like(v)
    _, next_state = gdn2_fix_step(state, q, k, v, alpha, erase, write, normalize_key=False)
    np.testing.assert_allclose(next_state, state + v[:, :, :, None] * k[:, :, None, :])


def test_zero_write_targeted_removal_does_not_globally_damp():
    state = np.zeros((1, 1, 2, 2), dtype=np.float32)
    state[0, 0, 0] = [4.0, 2.0]
    state[0, 0, 1] = [7.0, 3.0]
    q = np.ones((1, 1, 2), dtype=np.float32)
    k = np.array([[[1.0, 0.0]]], dtype=np.float32)
    alpha = np.ones_like(k)
    erase = np.ones_like(k)
    write = np.zeros((1, 1, 2), dtype=np.float32)
    _, next_state = gdn2_fix_step(state, q, k, np.zeros_like(write), alpha, erase, write, normalize_key=False)
    np.testing.assert_allclose(next_state[0, 0, :, 0], 0.0)
    np.testing.assert_allclose(next_state[0, 0, :, 1], state[0, 0, :, 1])


def test_repeated_key_overwrites_and_unrelated_key_is_preserved():
    rng = np.random.default_rng(4)
    query = rng.normal(size=(1, 3, 1, 2)).astype(np.float32)
    key = np.array([[[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]]], dtype=np.float32)
    value = np.array([[[[2.0, 0.0]], [[5.0, 0.0]], [[0.0, 7.0]]]], dtype=np.float32)
    alpha = np.ones_like(key)
    erase = np.ones_like(key)
    write = np.ones_like(value)
    _, state = gdn2_fix_scan(query, key, value, alpha, erase, write, normalize_key=False)
    np.testing.assert_allclose(state[0, 0, 0], [5.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(state[0, 0, 1], [0.0, 7.0], atol=1e-5)


def test_full_and_chunked_scans_match():
    rng = np.random.default_rng(9)
    arrays = [rng.normal(size=(2, 7, 3, width)).astype(np.float32) for width in (4, 4, 5, 4, 4, 5)]
    full = gdn2_fix_scan(*arrays)
    chunked = gdn2_fix_chunk_scan(*arrays, chunk_size=3)
    np.testing.assert_allclose(full[0], chunked[0], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(full[1], chunked[1], atol=1e-6, rtol=1e-6)


def test_torch_matches_numpy_forward_and_gradients_are_finite():
    rng = np.random.default_rng(2)
    values = [rng.normal(size=(1, 4, 2, width)).astype(np.float32) for width in (3, 3, 4, 3, 3, 4)]
    np_out, np_state = gdn2_fix_scan(*values)
    tensors = [torch.tensor(value, requires_grad=True) for value in values]
    torch_out, torch_state = torch_scan(*tensors)
    np.testing.assert_allclose(torch_out.detach().numpy(), np_out, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(torch_state.detach().numpy(), np_state, atol=1e-5, rtol=1e-5)
    (torch_out.square().mean() + torch_state.square().mean()).backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in tensors)


def test_numpy_forward_matches_finite_difference_gradient():
    rng = np.random.default_rng(11)
    arrays = [rng.normal(size=(1, 2, 1, width)).astype(np.float64) for width in (2, 2, 3, 2, 2, 3)]
    query, key, value, alpha, erase, write = arrays
    base_out, base_state = gdn2_fix_scan(query, key, value, alpha, erase, write)
    base_loss = float(np.sum(base_out * base_out) + np.sum(base_state * base_state))
    epsilon = 1e-6
    for array in (query, key, value, alpha, erase, write):
        index = (0, 0, 0, 0)
        original = array[index]
        array[index] = original + epsilon
        plus_out, plus_state = gdn2_fix_scan(query, key, value, alpha, erase, write)
        plus_loss = float(np.sum(plus_out * plus_out) + np.sum(plus_state * plus_state))
        array[index] = original - epsilon
        minus_out, minus_state = gdn2_fix_scan(query, key, value, alpha, erase, write)
        minus_loss = float(np.sum(minus_out * minus_out) + np.sum(minus_state * minus_state))
        array[index] = original
        numeric = (plus_loss - minus_loss) / (2 * epsilon)
        assert np.isfinite(numeric)
        assert np.isfinite(base_loss)
