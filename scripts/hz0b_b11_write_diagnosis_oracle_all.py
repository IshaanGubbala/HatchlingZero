"""HZ-0B B11 continuation: write-mechanism diagnosis, cell 1 of the
factorial matrix (`plans/HZ-0B_Progress_Tracker.md`, "Reopening
criteria").

| Write timing | Write content | Slot choice | Interpretation |
| --- | --- | --- | --- |
| **Oracle** | **Oracle** | **Oracle** | **Tests read path and memory substrate (this script)** |
| Oracle | Learned | Oracle | Tests content encoder |
| Learned | Oracle | Oracle | Tests write-trigger policy |
| Oracle | Oracle | Learned | Tests addressing |
| Learned | Learned | Learned | Full HZ-0B (already run, see hz0b_b11_baseline_comparison.py) |

This cell asks the most foundational question first: if the write is
handed to the model for free -- correct content, correct timing, correct
slot, nothing learned about WHETHER/WHAT/WHERE to write -- can the
mechanism even retrieve it? If this fails too, the problem is in the
read path or memory substrate itself, not the learned write controller.
If it succeeds, everything above (content encoder, write-trigger policy,
addressing) is implicated, not the substrate.

Reuses B6's own methodology (oracle write, train only the read path) on
B8 Stage 3's 2-way fact-discrimination task, at B11's reversal-triggering
scale (train_count=64, held_out_count=64) for direct comparability
against the already-measured 0.191 (soft), 0.269 (STE), and 0.512
(equal-param adapter) numbers.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from reference.hz0b_b6_hz0a_integration import frozen_hidden_states, logits_from_hidden
from reference.hz0b_memory_simulator import MemoryState
from reference.hz0b_readonly_integration import gated_memory_read, init_readonly_integration
from scripts.hz0b_b11_baseline_comparison import (
    D_MODEL, KEY_DIM, SEED, VALUE_DIM, load_frozen_model, make_prompts, targets_for,
)


def oracle_memory_state(is_a: mx.array, key_dim: int, value_dim: int) -> MemoryState:
    """1 slot, correct content for whichever fact this batch row actually
    showed -- the model never has to figure out WHAT to store or WHEN;
    it is handed the perfect write. `key_A`/`key_B` and `value_A`/
    `value_B` are fixed, distinguishable one-hot-scaled constants (same
    style as scripts/hz0b_generate_rust_parity_fixture.py's onehot
    convention), broadcast per-row by `is_a`."""
    batch = is_a.shape[0]
    key_a = mx.zeros((key_dim,)); key_a = mx.where(mx.arange(key_dim) == 0, mx.array(5.0), key_a)
    key_b = mx.zeros((key_dim,)); key_b = mx.where(mx.arange(key_dim) == 1, mx.array(5.0), key_b)
    value_a = mx.zeros((value_dim,)); value_a = mx.where(mx.arange(value_dim) == 0, mx.array(5.0), value_a)
    value_b = mx.zeros((value_dim,)); value_b = mx.where(mx.arange(value_dim) == 1, mx.array(5.0), value_b)

    key = mx.where(is_a[:, None] > 0.5, key_a[None, :], key_b[None, :])
    value = mx.where(is_a[:, None] > 0.5, value_a[None, :], value_b[None, :])
    return MemoryState(
        keys=key[:, None, :], values=value[:, None, :],
        confidence=mx.ones((batch, 1)), age=mx.zeros((batch, 1), dtype=mx.int32),
        protection=mx.zeros((batch, 1)), write_count=mx.ones((batch, 1), dtype=mx.int32),
        last_write_step=mx.zeros((batch, 1), dtype=mx.int32), write_source=mx.zeros((batch, 1), dtype=mx.int32),
    )


def run_oracle_all(model, train_tokens, train_is_a, held_out_tokens, held_out_is_a, *, seed: int, steps: int, lr: float) -> float:
    params = init_readonly_integration(D_MODEL, KEY_DIM, VALUE_DIM, seed=seed)
    params_dict = {"query_w": params.query_w, "query_b": params.query_b, "gate_w": params.gate_w,
                   "gate_b": params.gate_b, "value_to_hidden_w": params.value_to_hidden_w, "value_to_hidden_b": params.value_to_hidden_b}

    train_memory = oracle_memory_state(train_is_a, KEY_DIM, VALUE_DIM)
    held_out_memory = oracle_memory_state(held_out_is_a, KEY_DIM, VALUE_DIM)
    train_hidden, _ = frozen_hidden_states(model, train_tokens)
    held_out_hidden, _ = frozen_hidden_states(model, held_out_tokens)
    train_targets = targets_for(train_is_a)
    held_out_targets = targets_for(held_out_is_a)

    def loss_fn(pd: dict) -> mx.array:
        p = type(params)(**pd)
        final_hidden = train_hidden[:, -1, :]
        output, _ = gated_memory_read(p, final_hidden, train_memory)
        logits = logits_from_hidden(model, output[:, None, :])[:, -1, :]
        return mx.mean(nn.losses.cross_entropy(logits, train_targets))

    grad_fn = mx.value_and_grad(loss_fn)
    for step in range(steps):
        loss, grads = grad_fn(params_dict)
        mx.eval(loss)
        params_dict = {k: params_dict[k] - lr * grads[k] for k in params_dict}
        mx.eval(*params_dict.values())
        if step % 300 == 0 or step == steps - 1:
            print(f"    [oracle-all seed={seed}] step {step:4d}  train loss {float(loss):.5f}")

    trained = type(params)(**params_dict)
    held_out_final_hidden = held_out_hidden[:, -1, :]
    output, _ = gated_memory_read(trained, held_out_final_hidden, held_out_memory)
    logits = logits_from_hidden(model, output[:, None, :])[:, -1, :]
    predicted = mx.argmax(logits, axis=-1)
    return float(mx.mean((predicted == held_out_targets).astype(mx.float32)))


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

    rng = random.Random(SEED)
    train_tokens, train_is_a = make_prompts(args.train_count, rng)
    held_out_tokens, held_out_is_a = make_prompts(args.held_out_count, rng)
    print(f"train_count={args.train_count} held_out_count={args.held_out_count}")

    print(f"\nOracle-everything (timing=oracle, content=oracle, slot=oracle) -- tests read path + memory substrate ({args.num_seeds} seeds):")
    accs = []
    for seed_offset in range(args.num_seeds):
        acc = run_oracle_all(model, train_tokens, train_is_a, held_out_tokens, held_out_is_a, seed=SEED + seed_offset, steps=args.steps, lr=args.lr)
        print(f"  seed {SEED + seed_offset}: {acc:.3f}")
        accs.append(acc)
    mean = sum(accs) / len(accs)
    std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
    print(f"\n--- Summary ---\nmean: {mean:.3f}  std: {std:.3f}  range: {min(accs):.3f}-{max(accs):.3f}")
    print("\nReference points from the full investigation:")
    print("  floor (no memory, 0 params):        0.000")
    print("  soft-gate full HZ-0B (num_slots=8):  0.191")
    print("  STE full HZ-0B (num_slots=8):        0.269")
    print("  equal-param adapter (no memory):     0.512")
    if mean > 0.7:
        print("\nRESULT: oracle-everything memory decisively beats no-memory and the adapter -- "
              "the read path and memory substrate are FINE. The problem is entirely in the learned write controller.")
    elif mean > 0.512:
        print("\nRESULT: oracle-everything beats the adapter but not decisively -- read path/substrate are plausibly "
              "OK but not as strong a signal as hoped; the write controller is still clearly implicated but the "
              "substrate itself may also deserve scrutiny.")
    else:
        print("\nRESULT: oracle-everything does NOT beat the adapter -- the problem is NOT limited to the learned "
              "write controller. The read path or memory substrate itself is implicated.")


if __name__ == "__main__":
    main()
