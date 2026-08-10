import torch

from reference.hz0h_bdh_h8_interpretability import (
    _examples, _query_latents, run_h8_probe, train_concept_model,
)


def test_h8_trace_is_finite_and_shaped():
    model = train_concept_model(steps=20, batch_size=8, seed=1)
    xs, _ = _examples(2, count_per_concept=2)
    trace = _query_latents(model, xs[0])
    assert trace.shape == (model.config.n_head, model.config.mlp_internal_dim_multiplier * model.config.n_embd // model.config.n_head)
    assert torch.isfinite(trace).all()


def test_h8_probe_is_deterministic():
    a = run_h8_probe(seed=3, steps=30, top_k=4)
    b = run_h8_probe(seed=3, steps=30, top_k=4)
    assert a == b


def test_h8_reports_causal_ablation_result_without_claiming_a_win():
    result = run_h8_probe(seed=0, steps=100, top_k=4)
    assert 0 <= result.selected_ablation_accuracy <= 1
    assert 0 <= result.random_ablation_accuracy <= 1
    assert result.selectivity_margin >= 0
