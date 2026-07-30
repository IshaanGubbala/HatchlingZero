"""The decisive follow-up `docs/restart/hz0a_gdn3_tiny_lm_comparison_results.md`
recommended: generic real-text perplexity was tied, which doesn't test the
actual hypothesis (targeted overwrite matters) -- this does. Multi-query
associative recall WITH reassignment (the standard synthetic benchmark
family delta-net/linear-attention papers themselves use to demonstrate
this exact capability, e.g. MQAR-style tasks): several key->value
assignments, some keys get REASSIGNED (overwritten) later in the same
sequence, distractor tokens in between, then a query for one key's
CURRENT (most-recent) value.

If HZ-0A's GDN2 really has a structural overwrite weakness (confirmed at
the mechanism level in `docs/restart/hz0a_gdn3_overwrite_benchmark_results.md`),
this is where it should show up as an actual capability gap -- accuracy
on reassigned keys specifically, not generic perplexity.

Same tiny architecture as the prior LM comparison (dim=64, 4 layers, all
GDN-family, no attention), same fair-comparison discipline (identical
everything except the mixer). Trained from scratch on this synthetic task
(no real corpus pretraining) since the task itself, not language priors,
is what's being tested.
"""
from __future__ import annotations

import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from reference.hz0a_gdn3_tiny_lm import TinyGDNLM

VOCAB_SIZE, DIM, LAYERS, HEADS, D_FF = 512, 64, 4, 4, 128  # small synthetic vocab is enough for this task, faster to train
NUM_KEYS = 8            # distinct "variable" tokens: 10..17
NUM_VALUES = 8          # distinct "value" tokens: 20..27
DISTRACTOR_LOW, DISTRACTOR_HIGH = 100, 500  # irrelevant filler tokens, well clear of key/value/query ids
QUERY_TOKEN = 30
SEQ_LEN = 48
BATCH_SIZE = 32
STEPS, LR = 800, 3e-4
SEED = 999


def make_batch(rng: random.Random, batch_size: int) -> tuple[mx.array, mx.array]:
    """Returns (tokens [batch, SEQ_LEN], targets [batch] -- the correct,
    MOST-RECENT value for the queried key at the final position)."""
    rows, targets = [], []
    for _ in range(batch_size):
        current_value = {}
        events = []
        # 1. initial assignment for every key, in random order
        keys_order = list(range(NUM_KEYS))
        rng.shuffle(keys_order)
        for key in keys_order:
            value = rng.randrange(NUM_VALUES)
            current_value[key] = value
            events.append((10 + key, 20 + value))
        # 2. a few REASSIGNMENTS (overwrites) of randomly chosen keys, later in the sequence
        for _ in range(rng.randint(2, 4)):
            key = rng.randrange(NUM_KEYS)
            value = rng.randrange(NUM_VALUES)
            current_value[key] = value
            events.append((10 + key, 20 + value))
        rng.shuffle(events)  # interleave initial assignments and reassignments randomly, distractors added between all of them below

        row = []
        for key_tok, value_tok in events:
            row.append(rng.randint(DISTRACTOR_LOW, DISTRACTOR_HIGH))
            row.append(key_tok)
            row.append(value_tok)
        query_key = rng.randrange(NUM_KEYS)
        assert len(row) + 2 <= SEQ_LEN - 1, "events + query overflow SEQ_LEN -- would silently truncate the query cue, increase SEQ_LEN or reduce NUM_KEYS/reassignments"
        row.append(QUERY_TOKEN)
        row.append(10 + query_key)
        row += [rng.randint(DISTRACTOR_LOW, DISTRACTOR_HIGH) for _ in range(SEQ_LEN - 1 - len(row))]
        row.append(20 + current_value[query_key])  # final token IS the answer -- predicted from the second-to-last position
        rows.append(row)
        targets.append(20 + current_value[query_key])
    return mx.array(rows, dtype=mx.int32), mx.array(targets, dtype=mx.int32)


def loss_and_logits(model, tokens):
    logits, _ = model(tokens)
    final_logits = logits[:, -2, :]  # predicting the LAST token (the answer) from the second-to-last position
    targets = tokens[:, -1]
    loss = mx.mean(nn.losses.cross_entropy(final_logits, targets))
    return loss, final_logits, targets


def run(use_candidate: bool, seed: int) -> dict:
    label = "GDN-3 candidate" if use_candidate else "current GDN2"
    rng = random.Random(seed)
    mx.random.seed(seed)
    model = TinyGDNLM(VOCAB_SIZE, DIM, LAYERS, HEADS, D_FF, use_candidate)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=LR, weight_decay=0.01)

    def loss_fn(model, tokens):
        loss, _, _ = loss_and_logits(model, tokens)
        return loss

    value_and_grad = nn.value_and_grad(model, loss_fn)

    eval_tokens, eval_targets = make_batch(random.Random(seed + 1), 256)  # fixed held-out eval set, different seed
    mx.eval(eval_tokens, eval_targets)

    for step in range(STEPS):
        tokens, _ = make_batch(rng, BATCH_SIZE)
        loss, grads = value_and_grad(model, tokens)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)
        if step % 100 == 0 or step == STEPS - 1:
            _, eval_logits, _ = loss_and_logits(model, eval_tokens)
            mx.eval(eval_logits)
            predicted = mx.argmax(eval_logits, axis=-1)
            accuracy = float(mx.mean((predicted == eval_targets).astype(mx.float32)))
            print(f"[{label}] step {step:4d}  train_loss {float(loss):.4f}  eval_accuracy {accuracy:.4f}")

    _, eval_logits, _ = loss_and_logits(model, eval_tokens)
    mx.eval(eval_logits)
    predicted = mx.argmax(eval_logits, axis=-1)
    final_accuracy = float(mx.mean((predicted == eval_targets).astype(mx.float32)))
    return {"label": label, "final_accuracy": final_accuracy}


def main():
    print("=== current GDN2 ===")
    current_result = run(use_candidate=False, seed=SEED)
    print("\n=== GDN-3 candidate ===")
    candidate_result = run(use_candidate=True, seed=SEED)

    print("\n=== summary ===")
    print(f"current GDN2:      final eval accuracy {current_result['final_accuracy']:.4f}  (chance = {1.0/NUM_VALUES:.4f})")
    print(f"GDN-3 candidate:   final eval accuracy {candidate_result['final_accuracy']:.4f}  (chance = {1.0/NUM_VALUES:.4f})")
    diff = candidate_result['final_accuracy'] - current_result['final_accuracy']
    print(f"accuracy difference (candidate - current): {diff:+.4f}")


if __name__ == "__main__":
    main()
