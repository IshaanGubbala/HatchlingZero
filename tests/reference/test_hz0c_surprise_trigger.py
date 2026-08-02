"""HZ-0C C1/C2 regression tests: surprise-triggered anchor attention
(reference/hz0c_surprise_trigger.py)."""
from __future__ import annotations

import mlx.core as mx

from reference.hz0a_mlx_model import HZ0AMlxModel
from reference.hz0c_surprise_trigger import (
    HZ0CSurpriseTriggeredModel, SurpriseTriggeredBlock, masked_anchor_attention,
    surprise_score, trigger_decision,
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
