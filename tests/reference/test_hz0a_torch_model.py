import torch

from reference.hz0a_torch_model import HZ0AConfig, HZ0AModel, parameter_count


def test_configured_torch_model_runs_and_carries_recurrent_state():
    config = HZ0AConfig(32, 16, 3, 2, 8, 8, 32, (1,))
    model = HZ0AModel(config)
    tokens = torch.arange(10).reshape(1, 10) % config.vocab_size
    logits, states = model(tokens)
    assert logits.shape == (1, 10, config.vocab_size)
    assert states[0].shape == (1, 2, 8, 8)
    assert states[1] is None
    assert torch.isfinite(logits).all()
    _, carried = model(tokens[:, :5])
    second, _ = model(tokens[:, 5:], carried)
    assert second.shape == (1, 5, config.vocab_size)


def test_parameter_count_matches_module_parameters():
    config = HZ0AConfig(32, 16, 3, 2, 8, 8, 32, (1,))
    assert parameter_count(config) == sum(parameter.numel() for parameter in HZ0AModel(config).parameters())


def test_locked_hz0a_configuration_matches_a1_parameter_target_without_allocating_weights():
    config = HZ0AConfig.from_json("specs/hz0a_300m_a1.json")
    with torch.device("meta"):
        model = HZ0AModel(config)
    assert sum(parameter.numel() for parameter in model.parameters()) == 301_178_112
