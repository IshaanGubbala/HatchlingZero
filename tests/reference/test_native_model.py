import numpy as np
import torch

from restart.hz0a_pmetal.python.native_model import NativeTinyHZ0AModel
from restart.hz0a_pmetal.python.native_blocks import NativeTop1MoE
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


def test_native_model_moe_layer_routes_overflow_and_backpropagates():
    model = NativeTinyHZ0AModel(
        32, 16, 3, 2, 8, 8, 32, [1], seed=17,
        moe_layers=[1], moe_num_experts=2, moe_expert_d_ff=8, moe_capacity_factor=0.5,
    )
    tokens = np.arange(8).reshape(2, 4) % 32
    targets = np.roll(tokens, -1, axis=1)
    loss, _ = model.loss_and_backward(tokens, targets)
    assert np.isfinite(loss)
    assert all(np.isfinite(parameter.grad).all() for parameter in model.parameters())
    moe_parameters = model.blocks[1].mlp.parameters()
    assert any(np.linalg.norm(parameter.grad) > 0 for parameter in moe_parameters)
    assert model.blocks[1].mlp._cache[4].any()


def test_native_moe_backward_matches_finite_difference_away_from_route_boundary():
    moe = NativeTop1MoE("moe", 2, 2, 2, 2.0, np.random.default_rng(23))
    x = np.array([[0.2, -0.1], [0.4, 0.3]], dtype=np.float32)
    upstream = np.array([[0.7, -0.2], [0.1, 0.9]], dtype=np.float32)

    def objective():
        return float(np.sum(moe.forward(x) * upstream))

    moe.zero_grad()
    moe.forward(x)
    moe.backward(upstream)
    checks = [moe.router.bias, moe.experts[0].gate.weight, moe.fallback.down.weight]
    for parameter in checks:
        index = (0,) * parameter.data.ndim
        original = float(parameter.data[index])
        epsilon = 1e-3
        parameter.data[index] = original + epsilon
        plus = objective()
        parameter.data[index] = original - epsilon
        minus = objective()
        parameter.data[index] = original
        numerical = (plus - minus) / (2.0 * epsilon)
        assert np.isclose(parameter.grad[index], numerical, rtol=3e-2, atol=3e-3), (parameter.name, parameter.grad[index], numerical)


def test_native_moe_model_named_checkpoint_round_trip_is_exact():
    model = NativeTinyHZ0AModel(
        24, 8, 2, 2, 4, 4, 16, [1], seed=31,
        moe_layers=[1], moe_num_experts=2, moe_expert_d_ff=4,
    )
    checkpoint = model.state_dict()
    fingerprint = model.parameter_fingerprint()
    for parameter in model.parameters():
        parameter.data.fill(0.0)
    model.load_state_dict(checkpoint)
    assert model.parameter_fingerprint() == fingerprint
    assert set(model.state_dict()) == set(checkpoint)


def test_native_moe_finite_guards_reject_bad_inputs_and_gradients():
    moe = NativeTop1MoE("guarded", 2, 2, 2, 1.5, np.random.default_rng(5))
    with np.testing.assert_raises(FloatingPointError):
        moe.forward(np.array([[np.nan, 0.0]], dtype=np.float32))
    moe.forward(np.zeros((1, 2), dtype=np.float32))
    with np.testing.assert_raises(FloatingPointError):
        moe.backward(np.array([[np.inf, 0.0]], dtype=np.float32))


def test_native_moe_matches_torch_functional_oracle():
    torch.manual_seed(0)
    rng = np.random.default_rng(41)
    moe = NativeTop1MoE("oracle", 3, 2, 2, 0.5, rng, fallback_d_ff=4)
    x = np.array([[0.2, -0.3, 0.4], [0.1, 0.5, -0.2]], dtype=np.float32)
    upstream = np.array([[0.7, -0.2, 0.1], [-0.3, 0.4, 0.8]], dtype=np.float32)
    native_output = moe.forward(x)
    moe.backward(upstream)

    torch_params = {
        parameter.name: torch.tensor(parameter.data, dtype=torch.float32, requires_grad=True)
        for parameter in moe.parameters()
    }
    xt = torch.tensor(x)
    router = torch.nn.functional.linear(xt, torch_params["oracle.router.weight"], torch_params["oracle.router.bias"])
    probs = torch.softmax(router, dim=-1)
    chosen = torch.argmax(router, dim=-1)
    gate = probs[torch.arange(x.shape[0]), chosen]
    capacity = 1
    counts = [0, 0]
    overflow = []
    for expert in chosen.tolist():
        overflow.append(counts[expert] >= capacity)
        counts[expert] += 1
    overflow = torch.tensor(overflow, dtype=torch.bool)

    def swiglu(prefix, hidden):
        gate_value = torch.nn.functional.linear(xt, torch_params[f"oracle.{prefix}.gate.weight"], torch_params[f"oracle.{prefix}.gate.bias"])
        up_value = torch.nn.functional.linear(xt, torch_params[f"oracle.{prefix}.up.weight"], torch_params[f"oracle.{prefix}.up.bias"])
        product = torch.nn.functional.silu(gate_value) * up_value
        return torch.nn.functional.linear(product, torch_params[f"oracle.{prefix}.down.weight"], torch_params[f"oracle.{prefix}.down.bias"])

    oracle_output = torch.zeros_like(xt)
    for expert in range(2):
        expert_output = swiglu(f"experts.{expert}", 2)
        selected = (chosen == expert) & ~overflow
        oracle_output = oracle_output + expert_output * (selected * gate)[:, None]
    oracle_output = oracle_output + swiglu("fallback", 4) * overflow[:, None]
    oracle_loss = torch.sum(oracle_output * torch.tensor(upstream))
    oracle_loss.backward()
    np.testing.assert_allclose(native_output, oracle_output.detach().numpy(), rtol=2e-5, atol=2e-5)
    for parameter in moe.parameters():
        np.testing.assert_allclose(parameter.grad, torch_params[parameter.name].grad.numpy(), rtol=3e-4, atol=3e-5)
