"""HZ-0B B11: real-model version of B8 Stage 5's distractor-immunity
scenario -- the honest gap named repeatedly today (docs/restart/hz0b_b11_stage5_baseline_results.md,
docs/restart/hz0b_b8_stage5_results.md): Stage 5's own scenarios are
pure-simulator (oracle key/value written directly, no LM, no learned
write timing) -- this tests the REAL, LEARNED write mechanism (the
fixed lambda_sparse=0.1 controller) against a task with distractors
actually present as tokens in context, not just injected as raw
array writes.

Task: same 2-way fact discrimination as
scripts/hz0b_b11_baseline_comparison.py (FACT_MARKER + fact_id, delayed
recall, read trigger, predict TARGET_A/TARGET_B), but with
NUM_DISTRACTORS extra DISTRACTOR_MARKER + random-value pairs interspersed
in the middle section -- content that LOOKS marker-like (could plausibly
trigger the learned write gate) but must NOT overwrite or crowd out the
real fact's slot before the read. Distractor identity is irrelevant
(never queried), unlike the real fact.

Real, honest question this asks: does the FIXED (lambda_sparse=0.1)
learned write mechanism, which has never been tested with genuine
in-context distractors before, maintain its real accuracy when
distractor writes compete for the same limited slots?
"""
from __future__ import annotations

import argparse
import random

import mlx.core as mx
import mlx.nn as nn

from reference.hz0b_b6_hz0a_integration import frozen_hidden_states
from scripts.hz0b_b11_baseline_comparison import (
    FACT_A_ID, FACT_B_ID, FACT_MARKER, FACT_POS, MIDDLE_LEN, READ_TRIGGER_A,
    READ_TRIGGER_B, SEED, VOCAB_SIZE, load_frozen_model, run_hzb_memory,
)

DISTRACTOR_MARKER = 21100
NUM_DISTRACTORS = 4
# NOT scripts.hz0b_b11_baseline_comparison.LAMBDA_SPARSE -- that module's
# own constant is still the ORIGINAL buggy 5.0 (every prior script only
# ever overrode it via --lambda-sparse on the command line, never changed
# the constant itself). The validated fix is 0.1 -- hardcoded here
# directly, same convention scripts/hz0b_b11_passkey_task.py already uses.
LAMBDA_SPARSE = 0.1


def make_prompts_with_distractors(count: int, rng: random.Random) -> tuple[mx.array, mx.array]:
    """Same shape as hz0b_b11_baseline_comparison.make_prompts, plus
    NUM_DISTRACTORS extra (DISTRACTOR_MARKER, random_value) pairs
    scattered through the middle section -- real, in-context distractor
    writes a well-behaved mechanism must not let crowd out the real fact."""
    rows, fact_is_a = [], []
    for _ in range(count):
        prefix = [rng.randint(100, VOCAB_SIZE - 100) for _ in range(FACT_POS)]
        is_a = rng.random() < 0.5
        fact_id = FACT_A_ID if is_a else FACT_B_ID
        middle = [rng.randint(100, VOCAB_SIZE - 100) for _ in range(MIDDLE_LEN)]
        # scatter distractor marker+value pairs at evenly-spaced points in the middle
        distractor_positions = sorted(rng.sample(range(len(middle) - 1), NUM_DISTRACTORS))
        for offset, pos in enumerate(distractor_positions):
            insert_at = pos + offset * 2  # account for growing list as we insert
            distractor_value = rng.randint(100, VOCAB_SIZE - 100)
            middle[insert_at:insert_at] = [DISTRACTOR_MARKER, distractor_value]
        row = prefix + [FACT_MARKER, fact_id] + middle + [READ_TRIGGER_A, READ_TRIGGER_B]
        rows.append(row)
        fact_is_a.append(1.0 if is_a else 0.0)
    return mx.array(rows, dtype=mx.int32), mx.array(fact_is_a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--held-out-count", type=int, default=64)
    args = parser.parse_args()

    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")
    print(f"num_distractors={NUM_DISTRACTORS} lambda_sparse={LAMBDA_SPARSE}")

    rng = random.Random(SEED)
    train_tokens, train_is_a = make_prompts_with_distractors(args.train_count, rng)
    held_out_tokens, held_out_is_a = make_prompts_with_distractors(args.held_out_count, rng)
    print(f"train_count={args.train_count} held_out_count={args.held_out_count}")

    train_hidden, _ = frozen_hidden_states(model, train_tokens)
    held_out_hidden, _ = frozen_hidden_states(model, held_out_tokens)
    mx.eval(train_hidden, held_out_hidden)

    print(f"\nHZ-0B real memory WITH {NUM_DISTRACTORS} in-context distractors ({args.num_seeds} seeds):")
    accs = []
    for i in range(args.num_seeds):
        acc = run_hzb_memory(model, train_hidden, train_is_a, held_out_hidden, held_out_is_a, seed=SEED + i, steps=args.steps, lr=args.lr, lambda_sparse=LAMBDA_SPARSE)
        print(f"  seed {SEED + i}: {acc:.3f}")
        accs.append(acc)
    mean = sum(accs) / len(accs)
    std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
    print(f"\n--- Summary ---\nmean: {mean:.3f}  std: {std:.3f}  range: {min(accs):.3f}-{max(accs):.3f}")
    print("\nReference point (no distractors, same fixed mechanism, docs/restart/hz0b_b11_evaluation_results.md): "
          "mean 0.830 (std 0.173, 10 seeds)")
    if mean >= 0.75:
        print("\nRESULT: distractor-immunity holds up well for the real learned mechanism -- "
              "close to the no-distractor baseline.")
    elif mean >= 0.5:
        print("\nRESULT: distractors measurably hurt the real learned mechanism, but it still "
              "clearly beats chance -- a real, partial cost, not a collapse.")
    else:
        print("\nRESULT: distractors substantially degrade the real learned mechanism -- "
              "a real, previously-untested vulnerability in the fixed configuration.")


if __name__ == "__main__":
    main()
