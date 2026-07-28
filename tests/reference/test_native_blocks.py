import numpy as np
import torch

from restart.hz0a_pmetal.python.native_blocks import NativeRMSNorm, NativeSwiGLU, NativeSiLU, residual_backward, residual_forward


def test_native_rmsnorm_silu_residual_match_torch():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(2, 3, 8)).astype(np.float32)
    upstream = rng.normal(size=x.shape).astype(np.float32)
    norm = NativeRMSNorm("norm", 8)
    native = norm.forward(x)
    native_grad = norm.backward(upstream)
    tx = torch.tensor(x, requires_grad=True)
    tw = torch.ones(8, requires_grad=True)
    tout = tx * torch.rsqrt(tx.square().mean(-1, keepdim=True) + 1e-6) * tw
    (tout * torch.tensor(upstream)).sum().backward()
    np.testing.assert_allclose(native, tout.detach().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(native_grad, tx.grad.numpy(), rtol=2e-5, atol=2e-5)

    activation = NativeSiLU()
    ax = rng.normal(size=x.shape).astype(np.float32)
    ag = rng.normal(size=x.shape).astype(np.float32)
    np.testing.assert_allclose(activation.backward(ag) if False else activation.forward(ax), torch.nn.functional.silu(torch.tensor(ax)).numpy(), rtol=1e-6, atol=1e-6)
    activation.forward(ax)
    tax = torch.tensor(ax, requires_grad=True)
    (torch.nn.functional.silu(tax) * torch.tensor(ag)).sum().backward()
    np.testing.assert_allclose(activation.backward(ag), tax.grad.numpy(), rtol=1e-5, atol=1e-5)
    left, right = residual_backward(upstream)
    np.testing.assert_allclose(residual_forward(x, x), x + x)
    np.testing.assert_allclose(left, upstream)
    np.testing.assert_allclose(right, upstream)


def test_native_swiglu_matches_torch():
    rng = np.random.default_rng(8)
    block = NativeSwiGLU("mlp", 8, 12, rng)
    x = rng.normal(size=(2, 3, 8)).astype(np.float32)
    upstream = rng.normal(size=(2, 3, 8)).astype(np.float32)
    output = block.forward(x)
    grad_x = block.backward(upstream)
    gate = torch.tensor(block.gate.weight.data, requires_grad=True)
    gate_b = torch.tensor(block.gate.bias.data, requires_grad=True)
    up = torch.tensor(block.up.weight.data, requires_grad=True)
    up_b = torch.tensor(block.up.bias.data, requires_grad=True)
    down = torch.tensor(block.down.weight.data, requires_grad=True)
    down_b = torch.tensor(block.down.bias.data, requires_grad=True)
    tx = torch.tensor(x, requires_grad=True)
    tout = torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(tx, gate, gate_b)) * torch.nn.functional.linear(tx, up, up_b), down, down_b)
    (tout * torch.tensor(upstream)).sum().backward()
    np.testing.assert_allclose(output, tout.detach().numpy(), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(grad_x, tx.grad.numpy(), rtol=3e-5, atol=3e-5)
