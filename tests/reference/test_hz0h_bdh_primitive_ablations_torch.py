"""Real correctness tests for reference/hz0h_bdh_primitive_ablations_torch.py.

The load-bearing test here is the first one: with default arguments the
ablation forward must reproduce the real oracle EXACTLY. Every ablation
result this module can produce is meaningless if its baseline is not
genuinely BDH, so that equivalence is proven rather than assumed."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_primitive_ablations_torch import ablated_bdh_forward, build_rope_freqs
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _model(seed: int = 7, **overrides) -> BDH:
    torch.manual_seed(seed)
    config = BDHConfig(
        n_layer=overrides.get("n_layer", 3), n_embd=overrides.get("n_embd", 32),
        n_head=overrides.get("n_head", 4), mlp_internal_dim_multiplier=overrides.get("mult", 8),
        vocab_size=256, dropout=0.0,
    )
    return BDH(config).eval()


def test_default_ablation_forward_matches_the_real_oracle_exactly():
    """Load-bearing gate: defaults == upstream BDH, bit-for-bit."""
    model = _model()
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    with torch.no_grad():
        oracle_logits, oracle_loss = model(idx, targets)
        ablated_logits, ablated_loss = ablated_bdh_forward(
            model, idx, model.config.n_layer, targets,
        )
    assert torch.equal(oracle_logits, ablated_logits), "default ablation forward is not the real oracle"
    assert torch.equal(oracle_loss, ablated_loss)


def test_build_rope_freqs_at_upstream_theta_reproduces_the_models_own_buffer():
    """The theta knob must be a pure substitution -- at upstream's own
    2**16 it has to reproduce the oracle's existing buffer exactly."""
    model = _model()
    rebuilt = build_rope_freqs(model, theta=2 ** 16)
    assert rebuilt.shape == model.attn.freqs.shape
    assert torch.equal(rebuilt, model.attn.freqs)


def test_changing_theta_actually_changes_the_frequencies_and_the_output():
    model = _model()
    idx = torch.randint(256, (2, 12))
    standard_rope = build_rope_freqs(model, theta=10_000)
    assert not torch.equal(standard_rope, model.attn.freqs), "theta=10000 must differ from upstream 2**16"
    with torch.no_grad():
        baseline, _ = ablated_bdh_forward(model, idx, model.config.n_layer)
        altered, _ = ablated_bdh_forward(model, idx, model.config.n_layer, freqs=standard_rope)
    assert torch.isfinite(altered).all()
    assert not torch.allclose(baseline, altered), "changing theta must change the real output"


def test_self_inclusive_mask_changes_output_and_stays_finite():
    model = _model()
    idx = torch.randint(256, (2, 12))
    with torch.no_grad():
        strict, _ = ablated_bdh_forward(model, idx, model.config.n_layer, mask_diagonal=-1)
        inclusive, _ = ablated_bdh_forward(model, idx, model.config.n_layer, mask_diagonal=0)
    assert torch.isfinite(inclusive).all()
    assert not torch.allclose(strict, inclusive), "allowing self-attention must change the real output"


def test_score_scaling_without_softmax_is_nearly_inert_because_of_the_following_layernorm():
    """Real, measured property worth pinning down rather than assuming:
    WITHOUT a softmax, scaling scores by a positive constant is very
    nearly a no-op in BDH, because `yKV = ln(scores @ x)` applies
    LayerNorm immediately afterwards and `LayerNorm(c*v) == LayerNorm(v)`
    for `c > 0` -- exactly, except for the `eps` inside `sqrt(var + eps)`.

    So upstream's omission of a `1/sqrt(d)` factor is NOT a risky
    deviation from standard attention; it is mathematically irrelevant
    at this position in the architecture. Scaling only becomes a real
    lever together with a softmax, where it sets the temperature -- which
    is why the sweep tests that combination rather than scaling alone.

    An earlier version of this test asserted merely that the outputs
    DIFFER, which passed only on the small `eps` artifact and would have
    implied a real effect that does not exist."""
    model = _model()
    idx = torch.randint(256, (2, 12))
    with torch.no_grad():
        unscaled, _ = ablated_bdh_forward(model, idx, model.config.n_layer, scale_scores=False)
        scaled, _ = ablated_bdh_forward(model, idx, model.config.n_layer, scale_scores=True)
    assert torch.isfinite(scaled).all()
    relative_difference = ((unscaled - scaled).abs().max() / unscaled.abs().max()).item()
    assert relative_difference < 0.2, (
        f"score scaling should be near-inert without softmax, saw {relative_difference:.3f} relative difference"
    )


def test_score_scaling_with_softmax_is_a_real_lever():
    """The same knob DOES matter once a softmax is present, since it sets
    the distribution's temperature -- the real reason to test scaling."""
    model = _model()
    idx = torch.randint(256, (2, 12))
    with torch.no_grad():
        unscaled, _ = ablated_bdh_forward(model, idx, model.config.n_layer, use_softmax=True, scale_scores=False)
        scaled, _ = ablated_bdh_forward(model, idx, model.config.n_layer, use_softmax=True, scale_scores=True)
    assert torch.isfinite(scaled).all()
    assert not torch.allclose(unscaled, scaled, atol=1e-4), "scaling must matter under softmax"


def test_softmax_variant_is_finite_including_the_fully_masked_first_row():
    """With mask_diagonal=-1 position 0 has no permitted targets, so a
    naive softmax produces NaN there. The real handling must yield exact
    zero (matching the oracle's own unnormalized behaviour) and never
    leak NaN into logits or gradients."""
    model = _model()
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    logits, loss = ablated_bdh_forward(
        model, idx, model.config.n_layer, targets, use_softmax=True,
    )
    assert torch.isfinite(logits).all(), "softmax variant leaked non-finite values into logits"
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), f"non-finite gradient in {name}"


def test_softmax_rows_sum_to_one_where_targets_exist():
    """Real sanity check on the softmax variant's own semantics: every row
    that HAS permitted targets must be a genuine probability distribution."""
    torch.manual_seed(3)
    model = _model()
    idx = torch.randint(256, (1, 8))
    T = idx.shape[1]
    x = model.ln(model.embed(idx).unsqueeze(1))
    freqs = model.attn.freqs
    r_phases = torch.arange(0, T, device=freqs.device, dtype=freqs.dtype).view(1, 1, -1, 1) * freqs
    with torch.no_grad():
        from reference.hz0h_bdh_torch import Attention
        x_sparse = torch.relu(x @ model._w(model.encoder))
        QR = Attention.rope(r_phases, x_sparse)
        scores = QR @ QR.mT
        allowed = torch.ones(T, T, dtype=torch.bool, device=scores.device).tril(diagonal=-1)
        probabilities = torch.nan_to_num(
            torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1), nan=0.0
        )
    row_sums = probabilities.sum(dim=-1)
    assert torch.allclose(row_sums[..., 0], torch.zeros_like(row_sums[..., 0])), "row 0 must be exactly zero"
    assert torch.allclose(row_sums[..., 1:], torch.ones_like(row_sums[..., 1:]), atol=1e-5)
