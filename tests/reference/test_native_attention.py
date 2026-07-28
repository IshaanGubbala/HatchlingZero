import numpy as np
import torch

from restart.hz0a_pmetal.python.native_attention import NativeCausalAttention


def test_native_causal_attention_matches_torch_forward_backward():
    rng = np.random.default_rng(44)
    layer = NativeCausalAttention("attention", 8, 2, rng)
    x = rng.normal(size=(2, 4, 8)).astype(np.float32)
    upstream = rng.normal(size=x.shape).astype(np.float32)
    output = layer.forward(x)
    grad_x = layer.backward(upstream)
    qkv = torch.tensor(layer.qkv.weight.data, requires_grad=True)
    qkv_b = torch.tensor(layer.qkv.bias.data, requires_grad=True)
    out = torch.tensor(layer.out.weight.data, requires_grad=True)
    out_b = torch.tensor(layer.out.bias.data, requires_grad=True)
    tx = torch.tensor(x, requires_grad=True)
    packed = torch.nn.functional.linear(tx, qkv, qkv_b).reshape(2, 4, 2, 12)
    q, k, v = packed.chunk(3, dim=-1)
    scores = torch.einsum("bthd,bshd->bhts", q, k) / np.sqrt(4)
    mask = torch.triu(torch.full((4, 4), float("-inf")), 1)
    weights = torch.softmax(scores + mask, dim=-1)
    mixed = torch.einsum("bhts,bshd->bthd", weights, v).reshape(2, 4, 8)
    tout = torch.nn.functional.linear(mixed, out, out_b)
    (tout * torch.tensor(upstream)).sum().backward()
    np.testing.assert_allclose(output, tout.detach().numpy(), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(grad_x, tx.grad.numpy(), rtol=4e-5, atol=4e-5)
