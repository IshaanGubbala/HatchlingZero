import numpy as np
import torch

from restart.hz0a_pmetal.python.native_layers import NativeEmbedding, NativeLinear, NativeTiedLMHead, cross_entropy_backward, cross_entropy_forward


def test_native_embedding_tied_lm_head_cross_entropy_matches_torch():
    rng = np.random.default_rng(4)
    embedding = NativeEmbedding("embedding", 11, 5, rng)
    embedding.weight.data[:] = rng.normal(size=embedding.weight.data.shape).astype(np.float32)
    head = NativeTiedLMHead(embedding)
    ids = np.array([[1, 4, 2], [2, 1, 3]])
    targets = np.array([[4, 2, 1], [1, 3, 0]])
    hidden = embedding.forward(ids)
    logits = head.forward(hidden)
    loss, cache = cross_entropy_forward(logits, targets)
    grad_logits = cross_entropy_backward(cache)
    grad_hidden = head.backward(grad_logits)
    embedding.backward(grad_hidden)

    weight = torch.tensor(embedding.weight.data, requires_grad=True)
    torch_hidden = weight[torch.tensor(ids)]
    torch_logits = torch_hidden @ weight.T
    torch_loss = torch.nn.functional.cross_entropy(torch_logits.reshape(-1, 11), torch.tensor(targets).reshape(-1))
    torch_loss.backward()
    np.testing.assert_allclose(loss, float(torch_loss.detach()), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(embedding.weight.grad, weight.grad.detach().numpy(), rtol=2e-5, atol=2e-5)


def test_native_linear_forward_backward_matches_torch():
    rng = np.random.default_rng(7)
    layer = NativeLinear("linear", 4, 3, rng)
    x = rng.normal(size=(2, 3, 4)).astype(np.float32)
    upstream = rng.normal(size=(2, 3, 3)).astype(np.float32)
    output = layer.forward(x)
    grad_x = layer.backward(upstream)
    weight = torch.tensor(layer.weight.data, requires_grad=True)
    bias = torch.tensor(layer.bias.data, requires_grad=True)
    tx = torch.tensor(x, requires_grad=True)
    tout = tx @ weight.T + bias
    (tout * torch.tensor(upstream)).sum().backward()
    np.testing.assert_allclose(output, tout.detach().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(grad_x, tx.grad.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(layer.weight.grad, weight.grad.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(layer.bias.grad, bias.grad.numpy(), rtol=1e-5, atol=1e-5)
