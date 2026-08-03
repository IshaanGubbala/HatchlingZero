"""HZ-0C C2 exit gate: "surprise correlates with controlled novelty or
difficulty." Real evidence, not a unit-test sanity check --
`surprise_score` (`reference/hz0c_surprise_trigger.py`) computed on
REAL hidden states from the frozen HZ-0A checkpoint (same "real model,
controlled synthetic construction" pattern as every HZ-0B B11
real-model task this session), against sequences with a KNOWN novelty
position.

Scenario 1 (novelty point): a repeated 4-token cyclic pattern, with
ONE unexpected token injected at a known position, breaking the cycle.
Checks whether surprise_score is elevated AT and immediately AFTER the
injected token, vs. the steady-state repeated positions.

Scenario 2 (difficulty proxy): fully random (high-entropy, "hard to
predict") token sequences vs. a single constant token repeated (low-
entropy, "easy to predict") -- checks whether mean surprise differs in
the expected direction.
"""
from __future__ import annotations

import random

import mlx.core as mx

from reference.hz0a_mlx_model import HZ0AMlxModel
from reference.hz0b_b6_hz0a_integration import frozen_hidden_states
from reference.hz0c_surprise_trigger import normalize_score, rate_bounded_threshold, surprise_score
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, D_MODEL, D_FF, HEADS, LAYERS, VOCAB_SIZE, ATTENTION_INDICES
import json
from mlx.utils import tree_unflatten


def load_frozen_model():
    payload = json.loads((CHECKPOINT / "state.json").read_text())
    model = HZ0AMlxModel(VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF, ATTENTION_INDICES, native_metal=True)
    model_arrays = [(item["key"], mx.load(str(CHECKPOINT / item["file"]))) for item in payload["arrays"] if item["group"] == "model"]
    model.update(tree_unflatten(model_arrays))
    mx.eval(model.parameters())
    return model, payload


def make_novelty_sequences(count: int, rng: random.Random, *, pattern_reps: int = 8, pattern_len: int = 4) -> tuple[mx.array, list[int]]:
    """Returns (token_ids [count, seq_len], novelty_positions [count])."""
    rows, novelty_positions = [], []
    for _ in range(count):
        pattern = [rng.randint(100, VOCAB_SIZE - 200) for _ in range(pattern_len)]
        row = pattern * pattern_reps
        novelty_pos = rng.randrange(pattern_len * 2, pattern_len * (pattern_reps - 2))  # avoid the very start/end
        novelty_token = rng.randint(VOCAB_SIZE - 100, VOCAB_SIZE - 1)  # a distinct, unused-in-pattern range
        row[novelty_pos] = novelty_token
        rows.append(row)
        novelty_positions.append(novelty_pos)
    return mx.array(rows, dtype=mx.int32), novelty_positions


def main():
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")

    rng = random.Random(555)
    NUM_EXAMPLES = 32
    tokens, novelty_positions = make_novelty_sequences(NUM_EXAMPLES, rng)
    hidden, _ = frozen_hidden_states(model, tokens)
    mx.eval(hidden)

    raw_score = surprise_score(hidden)
    normed_score = normalize_score(raw_score, method="zscore")

    at_novelty, after_novelty, steady_state = [], [], []
    STARTUP_SKIP = 8  # skip the pattern's first 2 reps (startup transient, not yet steady-state)
    for i, pos in enumerate(novelty_positions):
        at_novelty.append(float(normed_score[i, pos]))
        if pos + 1 < normed_score.shape[1]:
            after_novelty.append(float(normed_score[i, pos + 1]))
        steady_mask = [t for t in range(STARTUP_SKIP, normed_score.shape[1]) if abs(t - pos) > 2]
        steady_state.extend(float(normed_score[i, t]) for t in steady_mask)

    mean_at = sum(at_novelty) / len(at_novelty)
    mean_after = sum(after_novelty) / len(after_novelty)
    mean_steady = sum(steady_state) / len(steady_state)

    print(f"\n--- Scenario 1: novelty point (n={NUM_EXAMPLES} sequences) ---")
    print(f"mean normalized surprise AT the injected novelty position: {mean_at:.3f}")
    print(f"mean normalized surprise immediately AFTER novelty:        {mean_after:.3f}")
    print(f"mean normalized surprise at steady-state (repeated) positions: {mean_steady:.3f}")
    print(f"delta (at novelty - steady state): {mean_at - mean_steady:+.3f}")
    print(f"delta (after novelty - steady state): {mean_after - mean_steady:+.3f}")

    fraction_novelty_above_steady = sum(1 for v in at_novelty if v > mean_steady) / len(at_novelty)
    print(f"fraction of examples where novelty position scores above the steady-state mean: {fraction_novelty_above_steady:.3f}")

    print(f"\n--- Scenario 2: difficulty proxy (random vs. constant tokens) ---")
    rng2 = random.Random(777)
    random_tokens = mx.array([[rng2.randint(100, VOCAB_SIZE - 100) for _ in range(32)] for _ in range(16)], dtype=mx.int32)
    constant_token = rng2.randint(100, VOCAB_SIZE - 100)
    constant_tokens = mx.array([[constant_token] * 32 for _ in range(16)], dtype=mx.int32)

    random_hidden, _ = frozen_hidden_states(model, random_tokens)
    constant_hidden, _ = frozen_hidden_states(model, constant_tokens)
    mx.eval(random_hidden, constant_hidden)

    random_score = surprise_score(random_hidden)
    constant_score = surprise_score(constant_hidden)
    mean_random = float(mx.mean(random_score[:, 2:]))  # skip early transient
    mean_constant = float(mx.mean(constant_score[:, 2:]))
    print(f"mean raw surprise, random (high-entropy) tokens: {mean_random:.4f}")
    print(f"mean raw surprise, constant (low-entropy) token: {mean_constant:.4f}")
    print(f"ratio (random / constant): {mean_random / max(mean_constant, 1e-8):.2f}x")

    print(f"\n--- Rate-bounded thresholding sanity (target_rate=0.15) ---")
    threshold = rate_bounded_threshold(normed_score, target_rate=0.15, min_rate=0.05, max_rate=0.5)
    triggered = normed_score > threshold
    achieved_rate = float(mx.mean(triggered.astype(mx.float32)))
    print(f"achieved trigger rate: {achieved_rate:.3f} (target 0.150)")
    novelty_triggered = sum(1 for i, pos in enumerate(novelty_positions) if bool(triggered[i, pos])) / NUM_EXAMPLES
    print(f"fraction of novelty positions that get triggered at this rate: {novelty_triggered:.3f}")


if __name__ == "__main__":
    main()
