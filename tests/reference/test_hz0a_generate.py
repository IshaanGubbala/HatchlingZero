import mlx.core as mx

from reference.hz0a_mlx_model import HZ0AMlxModel
from scripts.hz0a_generate import generate, sample_next


def _tiny_model():
    # dim=32/heads=2 -> head_dim=16, layers=4 with one attention layer (index 2)
    # mixed with gdn2_fix recurrent layers -- exercises both state types the
    # way generate() threads them, at a shape cheap enough for a unit test.
    return HZ0AMlxModel(vocab_size=64, dim=32, layers=4, heads=2, d_ff=64, attention_indices=(2,), native_metal=False, mixer="gdn2_fix")


def test_generate_produces_correct_length():
    model = _tiny_model()
    prompt_ids = [3, 5, 7]
    out = generate(model, prompt_ids, max_new_tokens=6, temperature=0.0, top_k=None, seed=0)
    assert len(out) == len(prompt_ids) + 6
    assert out[: len(prompt_ids)] == prompt_ids


def test_generate_greedy_is_deterministic():
    model = _tiny_model()
    prompt_ids = [1, 2, 3, 4]
    out1 = generate(model, prompt_ids, max_new_tokens=8, temperature=0.0, top_k=None, seed=0)
    out2 = generate(model, prompt_ids, max_new_tokens=8, temperature=0.0, top_k=None, seed=99)
    assert out1 == out2  # temperature=0.0 ignores seed entirely -- pure argmax


def test_generate_handles_mixed_attention_and_recurrent_layers():
    # The real point of this test: HZ0AMlxModel's states list mixes recurrent
    # state tensors and attention (k, v) cache tuples in one list, and
    # generate()'s incremental loop must thread both correctly across steps
    # without a shape mismatch or crash.
    model = _tiny_model()
    out = generate(model, [10, 20], max_new_tokens=5, temperature=0.0, top_k=None, seed=0)
    assert all(isinstance(t, int) and 0 <= t < 64 for t in out)


def test_sample_next_temperature_zero_matches_argmax():
    logits = mx.array([[1.0, 5.0, 2.0], [3.0, 1.0, 0.5]])
    tokens, _ = sample_next(logits, temperature=0.0, top_k=None, rng_key=mx.random.key(0))
    assert tokens.tolist() == [1, 0]


def test_sample_next_top_k_restricts_support():
    # With top_k=1 sampling degenerates to argmax regardless of temperature.
    logits = mx.array([[1.0, 5.0, 2.0, 0.0]])
    key = mx.random.key(0)
    for seed in range(5):
        key, sub = mx.random.split(key)
        tokens, _ = sample_next(logits, temperature=1.0, top_k=1, rng_key=sub)
        assert tokens.tolist() == [1]
