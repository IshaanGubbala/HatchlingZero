"""HZ-0C C2: follow-up to hz0c_c2_surprise_validation.py's negative
result. That script's Scenario 1 used a repeated pattern of RANDOM
token IDs (not real language) -- both `surprise_score` and
`state_novelty_score` failed to detect the injected anomaly, and
`docs/restart/hz0c_c2_surprise_validation_results.md` named a real,
disclosed hypothesis: random token-ID cycles may not form a genuine
"expectation" for a language-trained model to violate, so NEITHER
signal firing is consistent with a task-construction confound rather
than two bad signal choices.

This tests that hypothesis directly: instead of random token IDs, a
real 4-token n-gram is extracted from real packed training data
(`data/packed/repro_1024_val.jsonl`), repeated to form a genuine local
pattern, and the "anomaly" is ALSO a real, in-distribution token (from
a different random position in the real corpus) -- isolating whether
in-distribution-ness, not randomness, is what was missing.
"""
from __future__ import annotations

import json
import random

import mlx.core as mx

from reference.hz0b_b6_hz0a_integration import frozen_hidden_states
from reference.hz0c_surprise_trigger import normalize_score, rate_bounded_threshold, state_novelty_score, surprise_score
from scripts.hz0b_b11_baseline_comparison import load_frozen_model
from scripts.hz0c_c2_surprise_validation import evaluate_scenario1

DATA_PATH = "data/packed/repro_1024_val.jsonl"


def load_real_sequences(n: int) -> list[list[int]]:
    sequences = []
    with open(DATA_PATH) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            sequences.append(json.loads(line))
    return sequences


def make_natural_novelty_sequences(count: int, rng: random.Random, real_sequences: list[list[int]], *, pattern_len: int = 4, pattern_reps: int = 8) -> tuple[mx.array, list[int]]:
    rows, novelty_positions = [], []
    for _ in range(count):
        source = rng.choice(real_sequences)
        start = rng.randrange(0, len(source) - pattern_len)
        ngram = source[start:start + pattern_len]
        row = ngram * pattern_reps
        novelty_pos = rng.randrange(pattern_len * 2, pattern_len * (pattern_reps - 2))
        anomaly_source = rng.choice(real_sequences)
        anomaly_token = anomaly_source[rng.randrange(0, len(anomaly_source))]
        row[novelty_pos] = anomaly_token
        rows.append(row)
        novelty_positions.append(novelty_pos)
    return mx.array(rows, dtype=mx.int32), novelty_positions


def main():
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")

    real_sequences = load_real_sequences(200)
    print(f"loaded {len(real_sequences)} real packed sequences from {DATA_PATH}")

    rng = random.Random(555)
    NUM_EXAMPLES = 32
    tokens, novelty_positions = make_natural_novelty_sequences(NUM_EXAMPLES, rng, real_sequences)
    hidden, _ = frozen_hidden_states(model, tokens)
    mx.eval(hidden)

    evaluate_scenario1("delta-norm, REAL n-gram pattern + REAL anomaly token", surprise_score, hidden, novelty_positions, NUM_EXAMPLES)
    evaluate_scenario1("state-novelty w=4, REAL n-gram pattern + REAL anomaly token", lambda h: state_novelty_score(h, window=4), hidden, novelty_positions, NUM_EXAMPLES)
    evaluate_scenario1("state-novelty w=8, REAL n-gram pattern + REAL anomaly token", lambda h: state_novelty_score(h, window=8), hidden, novelty_positions, NUM_EXAMPLES)


if __name__ == "__main__":
    main()
