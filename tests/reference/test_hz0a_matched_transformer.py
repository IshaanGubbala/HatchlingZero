import torch

from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM, parameter_count


def test_matched_transformer_count_and_forward(tmp_path):
    values = {"vocab_size": 32, "d_model": 16, "num_layers": 2, "num_heads": 2, "head_dim": 8, "d_ff": 20, "tied_embeddings": True, "lm_head_bias": False}
    path = tmp_path / "transformer.json"
    path.write_text(__import__("json").dumps(values))
    config = MatchedTransformerConfig.from_json(path)
    model = MatchedTransformerLM(config)
    assert parameter_count(config) == sum(parameter.numel() for parameter in model.parameters())
    logits = model(torch.tensor([[1, 2, 3, 4]]))
    assert logits.shape == (1, 4, 32)
    assert bool(torch.isfinite(logits).all())


def test_locked_matched_transformer_count_is_exact():
    config = MatchedTransformerConfig.from_json("configs/hz0a_transformer_matched.json")
    assert parameter_count(config) == config.parameter_count == 301179928
