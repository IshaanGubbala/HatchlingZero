import numpy as np
import torch

from restart.hz0a_pmetal.python.native_model import NativeTinyHZ0AModel
from restart.hz0a_pmetal.python.training import PmetalOptimizerPath
from reference.hz0a_torch_model import HZ0AConfig, HZ0AModel


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


def test_native_complete_model_matches_torch_logits_loss_and_gradients():
    config = HZ0AConfig(32, 16, 3, 2, 8, 8, 32, (1,))
    torch_model = HZ0AModel(config)
    native = NativeTinyHZ0AModel(32, 16, 3, 2, 8, 8, 32, [1], seed=99)
    native_parameters = native.parameters()
    torch_parameters = [torch_model.embedding.weight]
    for block in torch_model.blocks:
        torch_parameters.append(block.norm1.weight)
        torch_parameters.extend([parameter for parameter in block.mixer.parameters()])
        torch_parameters.append(block.norm2.weight)
        torch_parameters.extend([parameter for parameter in block.mlp.parameters()])
    torch_parameters.append(torch_model.final_norm.weight)
    for native_parameter, torch_parameter in zip(native_parameters, torch_parameters):
        native_parameter.data[...] = torch_parameter.detach().numpy()
    tokens = np.arange(8).reshape(2, 4) % 32
    targets = np.roll(tokens, -1, axis=1)
    native_loss, _ = native.loss_and_backward(tokens, targets)
    torch_logits, _ = torch_model(torch.tensor(tokens))
    torch_loss = torch.nn.functional.cross_entropy(torch_logits.reshape(-1, 32), torch.tensor(targets).reshape(-1))
    torch_loss.backward()
    native_logits, _ = native.forward(tokens)
    np.testing.assert_allclose(native_logits, torch_logits.detach().numpy(), rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(native_loss, float(torch_loss.detach()), rtol=5e-4, atol=5e-4)
    for native_parameter, torch_parameter in zip(native_parameters, torch_parameters):
        np.testing.assert_allclose(native_parameter.grad, torch_parameter.grad.detach().numpy(), rtol=2e-3, atol=2e-3)


def test_native_complete_model_one_adamw_update_matches_torch():
    config = HZ0AConfig(32, 16, 3, 2, 8, 8, 32, (1,))
    torch_model = HZ0AModel(config)
    native = NativeTinyHZ0AModel(32, 16, 3, 2, 8, 8, 32, [1], seed=99)
    native_parameters = native.parameters()
    torch_parameters = [torch_model.embedding.weight]
    for block in torch_model.blocks:
        torch_parameters.append(block.norm1.weight); torch_parameters.extend(block.mixer.parameters()); torch_parameters.append(block.norm2.weight); torch_parameters.extend(block.mlp.parameters())
    torch_parameters.append(torch_model.final_norm.weight)
    for np_parameter, torch_parameter in zip(native_parameters, torch_parameters):
        np_parameter.data[...] = torch_parameter.detach().numpy()
    tokens = np.arange(8).reshape(2, 4) % 32; targets = np.roll(tokens, -1, axis=1)
    native.loss_and_backward(tokens, targets)
    native_optimizer = PmetalOptimizerPath(np.concatenate([p.data.reshape(-1) for p in native_parameters]).astype(np.float64), total_steps=1)
    native_optimizer.add_microbatch(np.concatenate([p.grad.reshape(-1) for p in native_parameters]).astype(np.float64), tokens=tokens.size)
    offset = 0
    for parameter in native_parameters:
        count = parameter.data.size; parameter.data[...] = native_optimizer.state.parameters[offset:offset + count].reshape(parameter.data.shape); offset += count
    torch_optimizer = torch.optim.AdamW(torch_model.parameters(), lr=1e-4, weight_decay=0.01)
    torch_optimizer.zero_grad(set_to_none=True)
    logits, _ = torch_model(torch.tensor(tokens))
    torch.nn.functional.cross_entropy(logits.reshape(-1, 32), torch.tensor(targets).reshape(-1)).backward()
    torch_optimizer.step()
    for np_parameter, torch_parameter in zip(native_parameters, torch_parameters):
        np.testing.assert_allclose(np_parameter.data, torch_parameter.detach().numpy(), rtol=3e-3, atol=3e-3)
