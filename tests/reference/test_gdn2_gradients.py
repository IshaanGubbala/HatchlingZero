from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def torch_gdn2_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay_logits: torch.Tensor,
    erase_logits: torch.Tensor,
    write_logits: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state
    outputs = []
    for t in range(q.shape[1]):
        decay = torch.sigmoid(decay_logits[:, t])
        erase = torch.sigmoid(erase_logits[:, t])
        write = torch.sigmoid(write_logits[:, t])
        decay_b = decay.unsqueeze(2)
        erase_b = erase.unsqueeze(2)
        write_b = write.unsqueeze(3)
        outer = v[:, t].unsqueeze(3) * k[:, t].unsqueeze(2)
        state = decay_b * ((1.0 - erase_b) * state) + write_b * outer
        outputs.append(torch.einsum("bhvk,bhk->bhv", state, q[:, t]))
    return torch.stack(outputs, dim=1), state


def torch_gdn2_chunk_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay_logits: torch.Tensor,
    erase_logits: torch.Tensor,
    write_logits: torch.Tensor,
    initial_state: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state
    outputs = []
    for start in range(0, q.shape[1], chunk_size):
        end = min(start + chunk_size, q.shape[1])
        out, state = torch_gdn2_scan(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            decay_logits[:, start:end],
            erase_logits[:, start:end],
            write_logits[:, start:end],
            state,
        )
        outputs.append(out)
    return torch.cat(outputs, dim=1), state


def manual_gdn2_backward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    decay_logits: np.ndarray,
    erase_logits: np.ndarray,
    write_logits: np.ndarray,
    initial_state: np.ndarray,
    grad_outputs: np.ndarray,
    grad_final_state: np.ndarray,
) -> dict[str, np.ndarray]:
    batch, steps, heads, d_k = q.shape
    d_v = v.shape[-1]

    states = [initial_state.astype(np.float64)]
    decays = []
    erases = []
    writes = []
    for t in range(steps):
        decay = 1.0 / (1.0 + np.exp(-np.clip(decay_logits[:, t], -30.0, 30.0)))
        erase = 1.0 / (1.0 + np.exp(-np.clip(erase_logits[:, t], -30.0, 30.0)))
        write = 1.0 / (1.0 + np.exp(-np.clip(write_logits[:, t], -30.0, 30.0)))
        decays.append(decay)
        erases.append(erase)
        writes.append(write)
        decay_b = decay[:, :, None, :]
        erase_b = erase[:, :, None, :]
        write_b = write[:, :, :, None]
        outer = v[:, t, :, :, None] * k[:, t, :, None, :]
        next_state = decay_b * ((1.0 - erase_b) * states[-1]) + write_b * outer
        states.append(next_state)

    grad_q = np.zeros_like(q, dtype=np.float64)
    grad_k = np.zeros_like(k, dtype=np.float64)
    grad_v = np.zeros_like(v, dtype=np.float64)
    grad_decay_logits = np.zeros_like(decay_logits, dtype=np.float64)
    grad_erase_logits = np.zeros_like(erase_logits, dtype=np.float64)
    grad_write_logits = np.zeros_like(write_logits, dtype=np.float64)

    grad_state = grad_final_state.astype(np.float64).copy()
    for t in reversed(range(steps)):
        state_t = states[t + 1]
        state_prev = states[t]
        q_t = q[:, t].astype(np.float64)
        k_t = k[:, t].astype(np.float64)
        v_t = v[:, t].astype(np.float64)
        decay = decays[t]
        erase = erases[t]
        write = writes[t]

        grad_y = grad_outputs[:, t].astype(np.float64)
        grad_q[:, t] = np.einsum("bhv,bhvk->bhk", grad_y, state_t)
        grad_state_total = grad_state + grad_y[:, :, :, None] * q_t[:, :, None, :]

        a = decay[:, :, None, :] * (1.0 - erase[:, :, None, :])
        grad_prev = grad_state_total * a

        grad_a = np.sum(grad_state_total * state_prev, axis=2)
        grad_decay = grad_a * (1.0 - erase)
        grad_erase = grad_a * (-decay)

        outer = v_t[:, :, :, None] * k_t[:, :, None, :]
        grad_write = np.sum(grad_state_total * outer, axis=3)
        grad_outer = grad_state_total * write[:, :, :, None]
        grad_v[:, t] = np.sum(grad_outer * k_t[:, :, None, :], axis=3)
        grad_k[:, t] = np.sum(grad_outer * v_t[:, :, :, None], axis=2)

        grad_decay_logits[:, t] = grad_decay * decay * (1.0 - decay)
        grad_erase_logits[:, t] = grad_erase * erase * (1.0 - erase)
        grad_write_logits[:, t] = grad_write * write * (1.0 - write)
        grad_state = grad_prev

    return {
        "q": grad_q,
        "k": grad_k,
        "v": grad_v,
        "decay_logits": grad_decay_logits,
        "erase_logits": grad_erase_logits,
        "write_logits": grad_write_logits,
        "initial_state": grad_state,
    }


def finite_difference_check() -> None:
    rng = np.random.default_rng(7)
    q = rng.normal(size=(1, 2, 1, 2)).astype(np.float64)
    k = rng.normal(size=(1, 2, 1, 2)).astype(np.float64)
    v = rng.normal(size=(1, 2, 1, 2)).astype(np.float64)
    decay = rng.normal(size=(1, 2, 1, 2)).astype(np.float64)
    erase = rng.normal(size=(1, 2, 1, 2)).astype(np.float64)
    write = rng.normal(size=(1, 2, 1, 2)).astype(np.float64)
    init = rng.normal(size=(1, 1, 2, 2)).astype(np.float64)
    grad_out = rng.normal(size=(1, 2, 1, 2)).astype(np.float64)
    grad_final = rng.normal(size=(1, 1, 2, 2)).astype(np.float64)
    manual = manual_gdn2_backward(q, k, v, decay, erase, write, init, grad_out, grad_final)

    eps = 1e-5

    def loss_fn(q_, k_, v_, d_, e_, w_, init_):
        qt = torch.tensor(q_, dtype=torch.float64)
        kt = torch.tensor(k_, dtype=torch.float64)
        vt = torch.tensor(v_, dtype=torch.float64)
        dt = torch.tensor(d_, dtype=torch.float64)
        et = torch.tensor(e_, dtype=torch.float64)
        wt = torch.tensor(w_, dtype=torch.float64)
        it = torch.tensor(init_, dtype=torch.float64)
        out, state = torch_gdn2_scan(qt, kt, vt, dt, et, wt, it)
        return float((out * torch.tensor(grad_out)).sum() + (state * torch.tensor(grad_final)).sum())

    numeric_q = np.zeros_like(q)
    for idx in np.ndindex(q.shape):
        q_pos = q.copy()
        q_neg = q.copy()
        q_pos[idx] += eps
        q_neg[idx] -= eps
        numeric_q[idx] = (loss_fn(q_pos, k, v, decay, erase, write, init) - loss_fn(q_neg, k, v, decay, erase, write, init)) / (2 * eps)
    np.testing.assert_allclose(manual["q"], numeric_q, atol=1e-5, rtol=1e-5)


def test_gdn2_manual_backward_matches_autodiff() -> None:
    torch.manual_seed(0)
    shapes = (2, 4, 2, 3, 2)
    batch, steps, heads, d_k, d_v = shapes
    q = torch.randn(batch, steps, heads, d_k, dtype=torch.float64, requires_grad=True)
    k = torch.randn(batch, steps, heads, d_k, dtype=torch.float64, requires_grad=True)
    v = torch.randn(batch, steps, heads, d_v, dtype=torch.float64, requires_grad=True)
    decay = torch.randn(batch, steps, heads, d_k, dtype=torch.float64, requires_grad=True)
    erase = torch.randn(batch, steps, heads, d_k, dtype=torch.float64, requires_grad=True)
    write = torch.randn(batch, steps, heads, d_v, dtype=torch.float64, requires_grad=True)
    init = torch.randn(batch, heads, d_v, d_k, dtype=torch.float64, requires_grad=True)

    out, state = torch_gdn2_scan(q, k, v, decay, erase, write, init)
    grad_out = torch.randn_like(out)
    grad_state = torch.randn_like(state)
    loss = (out * grad_out).sum() + (state * grad_state).sum()
    loss.backward()

    manual = manual_gdn2_backward(
        q.detach().numpy(),
        k.detach().numpy(),
        v.detach().numpy(),
        decay.detach().numpy(),
        erase.detach().numpy(),
        write.detach().numpy(),
        init.detach().numpy(),
        grad_out.detach().numpy(),
        grad_state.detach().numpy(),
    )

    np.testing.assert_allclose(manual["q"], q.grad.numpy(), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(manual["k"], k.grad.numpy(), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(manual["v"], v.grad.numpy(), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(manual["decay_logits"], decay.grad.numpy(), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(manual["erase_logits"], erase.grad.numpy(), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(manual["write_logits"], write.grad.numpy(), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(manual["initial_state"], init.grad.numpy(), atol=1e-8, rtol=1e-8)


def test_gdn2_manual_backward_handles_extreme_gate_values() -> None:
    torch.manual_seed(1)
    batch, steps, heads, d_k, d_v = 1, 3, 2, 2, 2
    q = torch.randn(batch, steps, heads, d_k, dtype=torch.float64, requires_grad=True)
    k = torch.randn(batch, steps, heads, d_k, dtype=torch.float64, requires_grad=True)
    v = torch.randn(batch, steps, heads, d_v, dtype=torch.float64, requires_grad=True)
    decay = torch.tensor([[[[8.0, -8.0], [6.0, -6.0]]] * steps], dtype=torch.float64, requires_grad=True)
    erase = torch.tensor([[[[-7.0, 7.0], [-5.0, 5.0]]] * steps], dtype=torch.float64, requires_grad=True)
    write = torch.tensor([[[[9.0, -9.0], [4.0, -4.0]]] * steps], dtype=torch.float64, requires_grad=True)
    init = torch.randn(batch, heads, d_v, d_k, dtype=torch.float64, requires_grad=True)

    out, state = torch_gdn2_scan(q, k, v, decay, erase, write, init)
    loss = out.square().sum() + state.square().sum()
    loss.backward()
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()
    assert torch.isfinite(decay.grad).all()
    assert torch.isfinite(erase.grad).all()
    assert torch.isfinite(write.grad).all()
    assert torch.isfinite(init.grad).all()


def test_gdn2_finite_difference_for_q() -> None:
    finite_difference_check()


def test_gdn2_manual_backward_matches_autodiff_across_lengths() -> None:
    torch.manual_seed(3)
    for steps in (1, 2, 5):
        q = torch.randn(1, steps, 2, 3, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, steps, 2, 3, dtype=torch.float64, requires_grad=True)
        v = torch.randn(1, steps, 2, 2, dtype=torch.float64, requires_grad=True)
        decay = torch.randn(1, steps, 2, 3, dtype=torch.float64, requires_grad=True)
        erase = torch.randn(1, steps, 2, 3, dtype=torch.float64, requires_grad=True)
        write = torch.randn(1, steps, 2, 2, dtype=torch.float64, requires_grad=True)
        init = torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=True)

        out, state = torch_gdn2_scan(q, k, v, decay, erase, write, init)
        grad_out = torch.randn_like(out)
        grad_state = torch.randn_like(state)
        loss = (out * grad_out).sum() + (state * grad_state).sum()
        loss.backward()

        manual = manual_gdn2_backward(
            q.detach().numpy(),
            k.detach().numpy(),
            v.detach().numpy(),
            decay.detach().numpy(),
            erase.detach().numpy(),
            write.detach().numpy(),
            init.detach().numpy(),
            grad_out.detach().numpy(),
            grad_state.detach().numpy(),
        )
        np.testing.assert_allclose(manual["initial_state"], init.grad.numpy(), atol=1e-8, rtol=1e-8)


def test_gdn2_chunked_gradients_match_full_scan() -> None:
    torch.manual_seed(4)
    q = torch.randn(1, 5, 2, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 5, 2, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 5, 2, 2, dtype=torch.float64, requires_grad=True)
    decay = torch.randn(1, 5, 2, 3, dtype=torch.float64, requires_grad=True)
    erase = torch.randn(1, 5, 2, 3, dtype=torch.float64, requires_grad=True)
    write = torch.randn(1, 5, 2, 2, dtype=torch.float64, requires_grad=True)
    init = torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=True)

    full_out, full_state = torch_gdn2_scan(q, k, v, decay, erase, write, init)
    chunk_out, chunk_state = torch_gdn2_chunk_scan(q, k, v, decay, erase, write, init, chunk_size=2)
    np.testing.assert_allclose(full_out.detach().numpy(), chunk_out.detach().numpy(), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(full_state.detach().numpy(), chunk_state.detach().numpy(), atol=1e-10, rtol=1e-10)

    full_loss = full_out.square().sum() + full_state.square().sum()
    full_loss.backward(retain_graph=True)
    full_q_grad = q.grad.detach().clone()
    q.grad.zero_()

    chunk_loss = chunk_out.square().sum() + chunk_state.square().sum()
    chunk_loss.backward()
    np.testing.assert_allclose(full_q_grad.numpy(), q.grad.numpy(), atol=1e-10, rtol=1e-10)
