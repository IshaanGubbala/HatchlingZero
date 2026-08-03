"""HZ-0C C4: fair anchor baselines. Per the plan: "Compare against no
anchors, fixed anchors, random anchors at matched rate, oracle
anchors, full attention, and an equal-compute transformer." Exit
gate: "quality can be compared at matched attention FLOPs."

Reuses C3's exact 8 real scenario constructions and ground-truth
positions (`scripts/hz0c_c3_trigger_simulator.py`) -- the real,
in-distribution content this project's own C2 lesson requires, not
synthetic random-ID tasks. Compares 6 trigger policies at a MATCHED
~15% activation rate (except no-anchor at 0%, full-attention at 100%,
and oracle which is exactly the ground-truth rate): no anchors, fixed
periodic, random (matched rate), oracle, `state_novelty_score`,
`token_loss_score`.

"Equal-compute transformer" is NOT built this pass -- it requires an
actual trained model at matched FLOPs, a much larger undertaking than
comparing trigger POLICIES on top of the same frozen backbone; named
explicitly as real, disclosed future work, not silently skipped.
"""
from __future__ import annotations

import random

import mlx.core as mx

from reference.hz0c_surprise_trigger import (
    fixed_periodic_trigger, full_attention_trigger, no_anchor_trigger, normalize_score, oracle_trigger,
    random_trigger, rate_bounded_threshold, state_novelty_score, token_loss_score,
)
from scripts.hz0b_b11_baseline_comparison import load_frozen_model
from scripts.hz0c_c3_trigger_simulator import (
    CODE_DATA_PATH, GENERAL_DATA_PATH, JSON_DATA_PATH, SEQ_LEN, TARGET_RATE, load_real_sequences,
    scenario_code_json_boundary, scenario_contradiction, scenario_distractor_heavy_retrieval,
    scenario_long_range_reappearance, scenario_rare_token_burst, scenario_repeated_pattern_anomaly,
    scenario_topic_shift, scenario_variable_rebinding,
)
from reference.hz0b_b6_hz0a_integration import frozen_hidden_states


def score_trigger(trigger: mx.array, gts: list[list[int]]) -> dict:
    seq_len = trigger.shape[1]
    tp = fp = fn = total_gt = total_positions = 0
    for i, gt_positions in enumerate(gts):
        gt_set = set(gt_positions)
        total_gt += len(gt_set)
        for t in range(seq_len):
            total_positions += 1
            is_triggered = bool(trigger[i, t] > 0.5)
            is_gt = t in gt_set
            if is_gt and is_triggered:
                tp += 1
            elif is_gt and not is_triggered:
                fn += 1
            elif not is_gt and is_triggered:
                fp += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    avg_rate = float(mx.mean(trigger))
    return {"precision": precision, "recall": recall, "avg_rate": avg_rate}


def evaluate_all_baselines(model, tokens: mx.array, gts: list[list[int]], seed: int) -> dict:
    hidden, _ = frozen_hidden_states(model, tokens)
    mx.eval(hidden)
    batch, seq_len = tokens.shape

    results = {}
    results["1. no_anchor"] = score_trigger(no_anchor_trigger(batch, seq_len), gts)
    results["2. fixed_periodic"] = score_trigger(fixed_periodic_trigger(batch, seq_len, period=8), gts)
    results["3. random_matched"] = score_trigger(random_trigger(batch, seq_len, rate=TARGET_RATE, seed=seed), gts)
    results["4. oracle"] = score_trigger(oracle_trigger(batch, seq_len, gts), gts)
    results["5. full_attention"] = score_trigger(full_attention_trigger(batch, seq_len), gts)

    state_novelty = normalize_score(state_novelty_score(hidden, window=4))
    state_novelty_trig = (state_novelty > rate_bounded_threshold(state_novelty, target_rate=TARGET_RATE, min_rate=0.02, max_rate=0.6)).astype(mx.float32)
    results["6. state_novelty (real-inference-safe)"] = score_trigger(state_novelty_trig, gts)

    token_loss = normalize_score(token_loss_score(model, hidden, tokens))
    token_loss_trig = (token_loss > rate_bounded_threshold(token_loss, target_rate=TARGET_RATE, min_rate=0.02, max_rate=0.6)).astype(mx.float32)
    results["7. token_loss (offline-only)"] = score_trigger(token_loss_trig, gts)

    return results


def main():
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")

    general = load_real_sequences(GENERAL_DATA_PATH, 200)
    code = load_real_sequences(CODE_DATA_PATH, 100)
    json_seqs = load_real_sequences(JSON_DATA_PATH, 100)
    print(f"loaded {len(general)} general, {len(code)} code, {len(json_seqs)} json real sequences\n")

    rng = random.Random(555)
    NUM_EXAMPLES = 32

    scenarios = {
        "1. Repeated pattern with anomaly": scenario_repeated_pattern_anomaly(NUM_EXAMPLES, rng, general),
        "2. Topic shift": scenario_topic_shift(NUM_EXAMPLES, rng, general),
        "3. Long-range key reappearance": scenario_long_range_reappearance(NUM_EXAMPLES, rng, general),
        "4. Changed variable bindings": scenario_variable_rebinding(NUM_EXAMPLES, rng, general),
        "5. Code/JSON boundary": scenario_code_json_boundary(NUM_EXAMPLES, rng, code, json_seqs),
        "6. Contradiction": scenario_contradiction(NUM_EXAMPLES, rng, general),
        "7. Rare-token burst": scenario_rare_token_burst(NUM_EXAMPLES, rng, general, 24576),
        "8. Distractor-heavy retrieval": scenario_distractor_heavy_retrieval(NUM_EXAMPLES, rng, general, code),
    }

    aggregate = {}
    for name, (tokens, gts) in scenarios.items():
        results = evaluate_all_baselines(model, tokens, gts, seed=555)
        print(f"--- {name} ---")
        for policy, r in results.items():
            print(f"  {policy:38s}  precision={r['precision']:.3f}  recall={r['recall']:.3f}  rate={r['avg_rate']:.3f}")
            aggregate.setdefault(policy, []).append(r["recall"])
        print()

    print("--- Mean recall across all 8 scenarios, per baseline ---")
    for policy, recalls in aggregate.items():
        mean_recall = sum(recalls) / len(recalls)
        print(f"  {policy:38s}  mean_recall={mean_recall:.3f}")


if __name__ == "__main__":
    main()
