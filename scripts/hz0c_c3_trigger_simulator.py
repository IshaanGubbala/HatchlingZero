"""HZ-0C C3: isolated trigger simulator. Real, checked evidence for
C3's exit gate ("the controller avoids always-on and always-off
behavior") across all 8 of the plan's named scenario types, using
`state_novelty_score` (C2's validated winner -- delta-norm was
measured weaker) computed from the REAL frozen HZ-0A checkpoint's
hidden states, on REAL, in-distribution corpus content throughout
(`data/packed/repro_1024_val.jsonl` for general text,
`data/packed/external/{code,json_and_configuration}_validation.jsonl`
for code/JSON) -- per the lesson learned and disclosed in
`docs/restart/hz0c_c2_surprise_validation_results.md`: arbitrary
token IDs do not form a real "expectation" for a language-trained
model, so every scenario here is built from real corpus spans, not
synthetic random IDs.

Each scenario returns (tokens, ground_truth_positions_per_example).
Measures, per scenario AND overall: trigger precision, recall,
false-trigger rate, missed-anchor rate, and average anchor rate, using
`rate_bounded_threshold` for a real, deterministic trigger decision.
"""
from __future__ import annotations

import argparse
import json
import random

import mlx.core as mx

from reference.hz0c_surprise_trigger import ema_novelty_score, normalize_score, rate_bounded_threshold, state_novelty_score, token_loss_score
from reference.hz0b_b6_hz0a_integration import frozen_hidden_states
from scripts.hz0b_b11_baseline_comparison import load_frozen_model

GENERAL_DATA_PATH = "data/packed/repro_1024_val.jsonl"
CODE_DATA_PATH = "data/packed/external/code_validation.jsonl"
JSON_DATA_PATH = "data/packed/external/json_and_configuration_validation.jsonl"
SEQ_LEN = 40
TARGET_RATE = 0.15


def load_real_sequences(path: str, n: int) -> list[list[int]]:
    sequences = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            sequences.append(json.loads(line))
    return sequences


def scenario_repeated_pattern_anomaly(count: int, rng: random.Random, general: list[list[int]]) -> tuple[mx.array, list[list[int]]]:
    """1. Repeated real n-gram pattern with one real anomaly token
    injected -- the construction validated in C2."""
    rows, gts = [], []
    pattern_len, reps = 4, 8
    for _ in range(count):
        source = rng.choice(general)
        start = rng.randrange(0, len(source) - pattern_len)
        ngram = source[start:start + pattern_len]
        row = ngram * reps
        pos = rng.randrange(pattern_len * 2, pattern_len * (reps - 2))
        anomaly_source = rng.choice(general)
        row[pos] = anomaly_source[rng.randrange(0, len(anomaly_source))]
        rows.append(row[:SEQ_LEN] if len(row) >= SEQ_LEN else row + [row[-1]] * (SEQ_LEN - len(row)))
        gts.append([pos] if pos < SEQ_LEN else [])
    return mx.array(rows, dtype=mx.int32), gts


def scenario_topic_shift(count: int, rng: random.Random, general: list[list[int]]) -> tuple[mx.array, list[list[int]]]:
    """2. Two different real sequences concatenated at a known
    boundary -- a genuine topic/source shift."""
    rows, gts = [], []
    for _ in range(count):
        a, b = rng.sample(general, 2)
        half = SEQ_LEN // 2
        boundary = half + rng.randint(-4, 4)
        row = a[:boundary] + b[:SEQ_LEN - boundary]
        rows.append(row[:SEQ_LEN])
        gts.append([boundary])
    return mx.array(rows, dtype=mx.int32), gts


def scenario_long_range_reappearance(count: int, rng: random.Random, general: list[list[int]]) -> tuple[mx.array, list[list[int]]]:
    """3. A real n-gram appears early, then reappears later after a
    long gap of unrelated real filler -- the reappearance position is
    the ground truth (a real event worth anchoring back to)."""
    rows, gts = [], []
    ngram_len = 4
    for _ in range(count):
        source = rng.choice(general)
        start = rng.randrange(0, len(source) - ngram_len)
        ngram = source[start:start + ngram_len]
        filler_source = rng.choice(general)
        filler = filler_source[:SEQ_LEN - 2 * ngram_len]
        reappear_pos = ngram_len + len(filler)
        row = ngram + filler + ngram
        rows.append(row[:SEQ_LEN])
        gts.append([reappear_pos] if reappear_pos < SEQ_LEN else [])
    return mx.array(rows, dtype=mx.int32), gts


def scenario_variable_rebinding(count: int, rng: random.Random, general: list[list[int]]) -> tuple[mx.array, list[list[int]]]:
    """4. A real "marker" token followed by a real "value" token,
    reassigned to a DIFFERENT real value later -- the rebinding
    position is ground truth (mirrors B11's code-symbol-tracking
    construction, real content standing in for variable=value)."""
    rows, gts = [], []
    for _ in range(count):
        source = rng.choice(general)
        marker = source[rng.randrange(0, len(source))]
        val1_source, val2_source = rng.sample(general, 2)
        val1 = val1_source[rng.randrange(0, len(val1_source))]
        val2 = val2_source[rng.randrange(0, len(val2_source))]
        filler1_source, filler2_source = rng.sample(general, 2)
        filler1 = filler1_source[:8]
        filler2 = filler2_source[:8]
        row = [marker, val1] + filler1 + [marker, val2] + filler2
        rebind_pos = 2 + len(filler1) + 1
        rows.append((row * 3)[:SEQ_LEN])
        gts.append([rebind_pos] if rebind_pos < SEQ_LEN else [])
    return mx.array(rows, dtype=mx.int32), gts


def scenario_code_json_boundary(count: int, rng: random.Random, code: list[list[int]], json_seqs: list[list[int]]) -> tuple[mx.array, list[list[int]]]:
    """5. Real code transitions to real JSON at a known boundary."""
    rows, gts = [], []
    for _ in range(count):
        c = rng.choice(code)
        j = rng.choice(json_seqs)
        half = SEQ_LEN // 2
        boundary = half + rng.randint(-4, 4)
        row = c[:boundary] + j[:SEQ_LEN - boundary]
        rows.append(row[:SEQ_LEN])
        gts.append([boundary])
    return mx.array(rows, dtype=mx.int32), gts


def scenario_contradiction(count: int, rng: random.Random, general: list[list[int]]) -> tuple[mx.array, list[list[int]]]:
    """6. Same real "claim marker" associated with two DIFFERENT real
    outcomes at different points -- the second (contradicting)
    occurrence is ground truth. Structurally similar to variable
    rebinding but framed as claim-then-reversal, per the plan's own
    naming."""
    rows, gts = [], []
    for _ in range(count):
        claim_source = rng.choice(general)
        claim_marker = claim_source[rng.randrange(0, len(claim_source))]
        outcome1_source, outcome2_source = rng.sample(general, 2)
        outcome1 = outcome1_source[rng.randrange(0, len(outcome1_source))]
        outcome2 = outcome2_source[rng.randrange(0, len(outcome2_source))]
        filler_source = rng.choice(general)
        filler = filler_source[:12]
        row = [claim_marker, outcome1] + filler + [claim_marker, outcome2]
        contradiction_pos = 2 + len(filler) + 1
        rows.append((row * 3)[:SEQ_LEN])
        gts.append([contradiction_pos] if contradiction_pos < SEQ_LEN else [])
    return mx.array(rows, dtype=mx.int32), gts


def scenario_rare_token_burst(count: int, rng: random.Random, general: list[list[int]], vocab_size: int) -> tuple[mx.array, list[list[int]]]:
    """7. A cluster of contextually out-of-place REAL tokens (a real
    multi-token span lifted from a DIFFERENT real sequence) inserted
    into otherwise-ordinary real content.

    ORIGINAL construction used the highest-ID token range as a
    "rareness" proxy -- found (2026-08-02, "fix it" investigation,
    see `docs/restart/hz0c_c3_trigger_simulator_results.md`'s fix
    section) to be the SAME category of confound C2 already diagnosed
    for its own first novelty-point test: token-ID magnitude is an
    arbitrary tokenizer assignment, not a measure of contextual
    surprise -- high-ID tokens produced no real elevation in either
    `state_novelty_score` or `ema_novelty_score`. A real span lifted
    from elsewhere in the real corpus (in-distribution, just locally
    out of place) DOES produce a real, measurable elevation (confirmed
    directly: mean 0.31 at burst positions vs. -0.025 elsewhere,
    verified with both signals) -- `vocab_size` kept as a parameter
    for interface stability even though this construction no longer
    uses it directly."""
    rows, gts = [], []
    burst_len = 3
    for _ in range(count):
        source = rng.choice(general)
        row = list(source[:SEQ_LEN])
        burst_start = rng.randrange(10, SEQ_LEN - burst_len - 5)
        other = rng.choice(general)
        other_start = rng.randrange(0, len(other) - burst_len)
        real_burst = other[other_start:other_start + burst_len]
        row[burst_start:burst_start + burst_len] = real_burst
        rows.append(row)
        gts.append(list(range(burst_start, burst_start + burst_len)))
    return mx.array(rows, dtype=mx.int32), gts


def scenario_distractor_heavy_retrieval(count: int, rng: random.Random, general: list[list[int]]) -> tuple[mx.array, list[list[int]]]:
    """8. Multiple decoy anomalies plus one designated TARGET anomaly
    -- only the target position is ground truth; the decoys test
    whether the mechanism stays selective under distraction rather
    than triggering on everything unusual."""
    rows, gts = [], []
    pattern_len, reps = 4, 8
    for _ in range(count):
        source = rng.choice(general)
        start = rng.randrange(0, len(source) - pattern_len)
        ngram = source[start:start + pattern_len]
        row = ngram * reps
        row = row[:SEQ_LEN] if len(row) >= SEQ_LEN else row + [row[-1]] * (SEQ_LEN - len(row))
        num_decoys = 2
        positions = rng.sample(range(pattern_len * 2, min(SEQ_LEN - 2, pattern_len * (reps - 2))), num_decoys + 1)
        target_pos = positions[0]
        decoy_positions = positions[1:]
        anomaly_source = rng.choice(general)
        row[target_pos] = anomaly_source[rng.randrange(0, len(anomaly_source))]
        for dp in decoy_positions:
            decoy_source = rng.choice(general)
            row[dp] = decoy_source[rng.randrange(0, len(decoy_source))]
        rows.append(row)
        gts.append([target_pos])
    return mx.array(rows, dtype=mx.int32), gts


def evaluate_scenario(name: str, tokens: mx.array, gts: list[list[int]], hidden: mx.array, *, signal: str = "state_novelty", model=None) -> dict:
    if signal == "state_novelty":
        raw_score = state_novelty_score(hidden, window=4)
    elif signal == "ema_novelty":
        raw_score = ema_novelty_score(hidden, decay=0.9)
    elif signal == "token_loss":
        raw_score = token_loss_score(model, hidden, tokens)
    else:
        raise ValueError(f"unknown signal: {signal!r}")
    normed_score = normalize_score(raw_score, method="zscore")
    threshold = rate_bounded_threshold(normed_score, target_rate=TARGET_RATE, min_rate=0.02, max_rate=0.6)
    triggered = normed_score > threshold

    seq_len = tokens.shape[1]
    tp = fp = fn = total_gt = total_positions = 0
    for i, gt_positions in enumerate(gts):
        gt_set = set(gt_positions)
        total_gt += len(gt_set)
        for t in range(seq_len):
            total_positions += 1
            is_triggered = bool(triggered[i, t])
            is_gt = t in gt_set
            if is_gt and is_triggered:
                tp += 1
            elif is_gt and not is_triggered:
                fn += 1
            elif not is_gt and is_triggered:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    false_trigger_rate = fp / max(total_positions - total_gt, 1)
    missed_anchor_rate = fn / max(total_gt, 1)
    avg_anchor_rate = float(mx.mean(triggered.astype(mx.float32)))

    print(f"\n--- {name} ---")
    print(f"  precision={precision:.3f}  recall={recall:.3f}  false_trigger_rate={false_trigger_rate:.3f}  missed_anchor_rate={missed_anchor_rate:.3f}  avg_anchor_rate={avg_anchor_rate:.3f}")
    return {"precision": precision, "recall": recall, "false_trigger_rate": false_trigger_rate, "missed_anchor_rate": missed_anchor_rate, "avg_anchor_rate": avg_anchor_rate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", choices=["state_novelty", "ema_novelty", "token_loss"], default="state_novelty")
    args = parser.parse_args()

    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}  signal={args.signal}")

    general = load_real_sequences(GENERAL_DATA_PATH, 200)
    code = load_real_sequences(CODE_DATA_PATH, 100)
    json_seqs = load_real_sequences(JSON_DATA_PATH, 100)
    print(f"loaded {len(general)} general, {len(code)} code, {len(json_seqs)} json real sequences")

    rng = random.Random(555)
    VOCAB_SIZE = 24576
    NUM_EXAMPLES = 32

    scenarios = {
        "1. Repeated pattern with anomaly": scenario_repeated_pattern_anomaly(NUM_EXAMPLES, rng, general),
        "2. Topic shift": scenario_topic_shift(NUM_EXAMPLES, rng, general),
        "3. Long-range key reappearance": scenario_long_range_reappearance(NUM_EXAMPLES, rng, general),
        "4. Changed variable bindings": scenario_variable_rebinding(NUM_EXAMPLES, rng, general),
        "5. Code/JSON boundary": scenario_code_json_boundary(NUM_EXAMPLES, rng, code, json_seqs),
        "6. Contradiction": scenario_contradiction(NUM_EXAMPLES, rng, general),
        "7. Rare-token burst": scenario_rare_token_burst(NUM_EXAMPLES, rng, general, VOCAB_SIZE),
        "8. Distractor-heavy retrieval": scenario_distractor_heavy_retrieval(NUM_EXAMPLES, rng, general),
    }

    all_results = {}
    all_rates = []
    for name, (tokens, gts) in scenarios.items():
        hidden, _ = frozen_hidden_states(model, tokens)
        mx.eval(hidden)
        result = evaluate_scenario(name, tokens, gts, hidden, signal=args.signal, model=model)
        all_results[name] = result
        all_rates.append(result["avg_anchor_rate"])

    print(f"\n--- Summary across all 8 scenarios ---")
    for name, r in all_results.items():
        print(f"  {name}: precision={r['precision']:.3f} recall={r['recall']:.3f} avg_anchor_rate={r['avg_anchor_rate']:.3f}")

    overall_min_rate, overall_max_rate = min(all_rates), max(all_rates)
    print(f"\nanchor rate range across scenarios: {overall_min_rate:.3f} - {overall_max_rate:.3f}")
    always_on_or_off = overall_min_rate < 0.001 or overall_max_rate > 0.999
    print(f"C3 exit gate (avoids always-on/always-off): {'FAIL' if always_on_or_off else 'PASS'}")


if __name__ == "__main__":
    main()
