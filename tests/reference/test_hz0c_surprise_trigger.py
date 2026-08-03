"""HZ-0C C1/C2 regression tests: surprise-triggered anchor attention
(reference/hz0c_surprise_trigger.py)."""
from __future__ import annotations

import mlx.core as mx

from reference.hz0a_mlx_model import HZ0AMlxModel
from reference.hz0c_surprise_trigger import (
    HZ0CSurpriseTriggeredModel, SurpriseTriggeredBlock, masked_anchor_attention,
    normalize_score, rate_bounded_threshold, smooth_score, state_novelty_score, surprise_score, trigger_decision,
)


def test_surprise_score_first_position_is_zero():
    hidden = mx.random.normal((2, 5, 16))
    score = surprise_score(hidden)
    assert score.shape == (2, 5)
    assert bool(mx.all(score[:, 0] == 0.0))


def test_surprise_score_zero_for_constant_hidden_state():
    """Epsilon-stabilized sqrt (`sqrt(0 + 1e-8)`) floors the score at
    ~1e-4 rather than exact 0 for a truly unchanging hidden state --
    same epsilon-under-sqrt pattern as
    reference/hz0b_memory_simulator.py's cosine similarity, for the
    same differentiability reason. Checked against that floor, not
    exact zero."""
    hidden = mx.ones((1, 4, 8))
    score = surprise_score(hidden)
    assert bool(mx.all(mx.abs(score) < 2e-4))


def test_surprise_score_positive_for_changing_hidden_state():
    hidden = mx.stack([mx.zeros((1, 8))[0], mx.ones((1, 8))[0] * 5.0], axis=0)[None]
    score = surprise_score(hidden)
    assert float(score[0, 1]) > 1.0


def test_trigger_decision_soft_is_in_unit_interval():
    score = mx.array([[0.0, 1.0, 5.0, -3.0]])
    trigger = trigger_decision(score, scale=mx.array([1.0]), bias=mx.array([0.0]))
    assert bool(mx.all(trigger > 0.0)) and bool(mx.all(trigger < 1.0))


def test_trigger_decision_ste_is_hard_zero_or_one():
    score = mx.array([[0.0, 10.0, -10.0]])
    trigger = trigger_decision(score, scale=mx.array([1.0]), bias=mx.array([0.0]), ste=True)
    for v in trigger[0].tolist():
        assert v in (0.0, 1.0)


def test_masked_anchor_attention_zero_at_nontriggered_query_positions():
    dim, heads, seq = 16, 2, 4
    x = mx.random.normal((1, seq, dim))
    trigger = mx.array([[1.0, 0.0, 1.0, 0.0]])
    qkv_w = mx.random.normal((3 * dim, dim)) * 0.1
    qkv_b = mx.zeros((3 * dim,))
    out_w = mx.random.normal((dim, dim)) * 0.1
    out_b = mx.zeros((dim,))
    out = masked_anchor_attention(x, trigger, qkv_w=qkv_w, qkv_b=qkv_b, out_w=out_w, out_b=out_b, heads=heads)
    assert bool(mx.all(out[:, 1, :] == 0.0))
    assert bool(mx.all(out[:, 3, :] == 0.0))


def test_masked_anchor_attention_only_attends_to_triggered_keys():
    """A triggered query position's output must depend only on
    triggered keys/values -- verified by checking it is UNCHANGED when
    a non-triggered position's value is perturbed."""
    dim, heads, seq = 16, 2, 5
    x = mx.random.normal((1, seq, dim))
    trigger = mx.array([[1.0, 0.0, 1.0, 0.0, 1.0]])
    qkv_w = mx.random.normal((3 * dim, dim)) * 0.1
    qkv_b = mx.zeros((3 * dim,))
    out_w = mx.random.normal((dim, dim)) * 0.1
    out_b = mx.zeros((dim,))
    out1 = masked_anchor_attention(x, trigger, qkv_w=qkv_w, qkv_b=qkv_b, out_w=out_w, out_b=out_b, heads=heads)

    x_perturbed = x.tolist()
    x_perturbed[0][1] = [v + 100.0 for v in x_perturbed[0][1]]  # perturb non-triggered position 1
    x2 = mx.array(x_perturbed)
    out2 = masked_anchor_attention(x2, trigger, qkv_w=qkv_w, qkv_b=qkv_b, out_w=out_w, out_b=out_b, heads=heads)
    assert bool(mx.allclose(out1[:, 4, :], out2[:, 4, :], atol=1e-4))


def test_surprise_triggered_block_forward_finite():
    block = SurpriseTriggeredBlock(dim=32, heads=4, d_ff=64)
    x = mx.random.normal((2, 8, 32))
    out, state = block(x)
    assert out.shape == x.shape
    assert bool(mx.all(mx.isfinite(out)))


def test_surprise_triggered_block_gradients_finite():
    block = SurpriseTriggeredBlock(dim=32, heads=4, d_ff=64)
    mx.eval(block.parameters())
    x = mx.random.normal((2, 8, 32))

    def loss_fn(params):
        block.update(params)
        out, _ = block(x)
        return mx.mean(out * out)

    grad_fn = mx.value_and_grad(loss_fn)
    loss, grads = grad_fn(block.trainable_parameters())
    assert bool(mx.isfinite(loss))

    def check(t):
        if isinstance(t, mx.array):
            return bool(mx.all(mx.isfinite(t)))
        if isinstance(t, dict):
            return all(check(v) for v in t.values())
        if isinstance(t, list):
            return all(check(v) for v in t)
        return True

    assert check(grads)


def test_c1_three_models_have_audited_parameter_counts():
    """Locks in C1's real, audited parameter counts at the actual
    301M-scale topology used throughout HZ-0B, matching the frozen
    checkpoint's own model 2 count exactly (cross-validated against
    plans/GDN-2_Fix.md's cited 301,178,112)."""
    from mlx.utils import tree_flatten

    vocab_size, dim, layers, heads, d_ff = 24576, 768, 31, 12, 2304
    anchor_indices = (4, 9, 14, 19, 24, 29)

    def count(model):
        return sum(v.size for _, v in tree_flatten(model.parameters()))

    model1 = HZ0AMlxModel(vocab_size, dim, layers, heads, d_ff, attention_indices=())
    model2 = HZ0AMlxModel(vocab_size, dim, layers, heads, d_ff, attention_indices=anchor_indices)
    model3 = HZ0CSurpriseTriggeredModel(vocab_size, dim, layers, heads, d_ff, anchor_indices=anchor_indices)

    assert count(model2) == 301_178_112  # the real, already-frozen checkpoint's own count
    assert count(model1) > count(model2)  # recurrent layers cost more params than attention here
    assert count(model3) > count(model1)  # anchor-capable layers pay for BOTH recurrence and attention


def test_c1_three_models_forward_pass_on_same_real_tokens():
    vocab_size, dim, layers, heads, d_ff = 24576, 64, 4, 4, 128
    anchor_indices = (1, 3)
    model1 = HZ0AMlxModel(vocab_size, dim, layers, heads, d_ff, attention_indices=())
    model2 = HZ0AMlxModel(vocab_size, dim, layers, heads, d_ff, attention_indices=anchor_indices)
    model3 = HZ0CSurpriseTriggeredModel(vocab_size, dim, layers, heads, d_ff, anchor_indices=anchor_indices)
    tokens = mx.array([[1, 2, 3, 4, 5, 6]])
    for model in (model1, model2, model3):
        logits, _ = model(tokens)
        assert logits.shape == (1, 6, vocab_size)
        assert bool(mx.all(mx.isfinite(logits)))


def test_normalize_score_zscore_has_zero_mean_unit_std():
    score = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    normed = normalize_score(score, method="zscore")
    assert abs(float(mx.mean(normed))) < 1e-4
    assert abs(float(mx.std(normed)) - 1.0) < 1e-3


def test_normalize_score_minmax_bounds_zero_one():
    score = mx.array([[1.0, 5.0, 3.0, -2.0]])
    normed = normalize_score(score, method="minmax")
    assert abs(float(mx.min(normed))) < 1e-5
    assert abs(float(mx.max(normed)) - 1.0) < 1e-5


def test_normalize_score_per_row_not_across_batch():
    """Two rows with different raw scales must each normalize to their
    OWN zero-mean/unit-std, not be normalized jointly."""
    score = mx.array([[0.0, 10.0], [100.0, 200.0]])
    normed = normalize_score(score, method="zscore")
    for row in range(2):
        assert abs(float(mx.mean(normed[row]))) < 1e-4


def test_smooth_score_identity_at_window_one():
    score = mx.array([[1.0, 5.0, 2.0, 9.0]])
    assert bool(mx.array_equal(smooth_score(score, window=1), score))


def test_smooth_score_averages_causally_no_future_leakage():
    score = mx.array([[10.0, 0.0, 0.0, 0.0]])
    smoothed = smooth_score(score, window=2)
    # position 0: avg of just itself (10). position 1: avg(10,0)=5.
    assert abs(float(smoothed[0, 0]) - 10.0) < 1e-5
    assert abs(float(smoothed[0, 1]) - 5.0) < 1e-5
    # position 2 must NOT see position 0's spike (window=2 excludes it) -- no future or stale leakage beyond the window
    assert abs(float(smoothed[0, 2]) - 0.0) < 1e-5


def test_rate_bounded_threshold_achieves_target_rate():
    mx.random.seed(3)
    score = mx.random.normal((1, 1000))
    threshold = rate_bounded_threshold(score, target_rate=0.1, min_rate=0.01, max_rate=0.5)
    achieved_rate = float(mx.mean((score > threshold).astype(mx.float32)))
    assert abs(achieved_rate - 0.1) < 0.02


def test_rate_bounded_threshold_clamps_to_bounds():
    mx.random.seed(4)
    score = mx.random.normal((1, 1000))
    threshold = rate_bounded_threshold(score, target_rate=0.9, min_rate=0.01, max_rate=0.2)
    achieved_rate = float(mx.mean((score > threshold).astype(mx.float32)))
    assert achieved_rate <= 0.25  # clamped toward max_rate=0.2, not the requested 0.9


def test_rate_bounded_threshold_deterministic():
    mx.random.seed(5)
    score = mx.random.normal((2, 50))
    t1 = rate_bounded_threshold(score, target_rate=0.2, min_rate=0.01, max_rate=0.5)
    t2 = rate_bounded_threshold(score, target_rate=0.2, min_rate=0.01, max_rate=0.5)
    assert bool(mx.array_equal(t1, t2))


def test_state_novelty_score_first_position_is_zero():
    hidden = mx.random.normal((2, 5, 8))
    score = state_novelty_score(hidden)
    assert bool(mx.all(score[:, 0] == 0.0))


def test_state_novelty_score_zero_for_constant_hidden_state():
    hidden = mx.ones((1, 6, 8))
    score = state_novelty_score(hidden)
    assert bool(mx.all(mx.abs(score) < 1e-4))


def test_state_novelty_score_high_for_pattern_break():
    """A repeating 2-position pattern [A, B, A, B, ...] with one
    anomalous vector C inserted should score C's position much higher
    than the steady-state A/B positions -- the exact case
    surprise_score (delta norm) was found to fail on."""
    a = mx.array([1.0, 0.0, 0.0, 0.0])
    b = mx.array([0.0, 1.0, 0.0, 0.0])
    c = mx.array([0.0, 0.0, 0.0, -1.0])  # orthogonal to both a and b
    seq = mx.stack([a, b, a, b, a, b, c, a, b, a, b], axis=0)[None]
    score = state_novelty_score(seq, window=4)
    anomaly_score = float(score[0, 6])
    steady_scores = [float(score[0, t]) for t in (4, 5, 8, 9, 10)]
    mean_steady = sum(steady_scores) / len(steady_scores)
    assert anomaly_score > mean_steady
