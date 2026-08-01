"""HZ-0B B11 continuation: write-mechanism diagnosis, factorial cell 2.

| Write timing | Write content | Slot choice | Interpretation |
| --- | --- | --- | --- |
| Oracle | Oracle | Oracle | Tests read path and memory substrate (run: mean 0.306) |
| **Oracle** | **Learned** | **Oracle** | **Tests content encoder (this script)** |
| Learned | Oracle | Oracle | Tests write-trigger policy (run: mean 0.053, WORSE than either extreme -- see docs/restart/hz0b_b11_evaluation_results.md, "the naive expectation is wrong") |
| Oracle | Oracle | Learned | Tests addressing (not run yet) |
| Learned | Learned | Learned | Full HZ-0B (soft 0.191, STE 0.269) |

Cell 3 found that isolating write-TIMING learning, with position-
invariant oracle content, starves the gate of gradient signal and
collapses to "never write" for most seeds. This cell avoids that
specific pathology by construction: timing is ORACLE (a single write is
always committed at a known, fixed position -- no gate to collapse),
while CONTENT is learned (key_proj/value_proj must learn to encode which
fact was actually shown into a key/value pair the read path can later
discriminate). Slot is a single oracle slot, same as every other cell.

Write position is fixed at `FACT_POS + 1` -- one position after the
fact-id token, the earliest point at which the fact is knowable, mirroring
B7's own oracle-timing convention (`WRITE_POS` in
`scripts/hz0b_b7_real_integration_probe.py`).
"""
from __future__ import annotations

import argparse
import random

import mlx.core as mx
import mlx.nn as nn

from reference.hz0b_b6_hz0a_integration import frozen_hidden_states, logits_from_hidden
from reference.hz0b_memory_simulator import MemoryState
from reference.hz0b_memory_simulator import write as memory_write
from reference.hz0b_readonly_integration import ReadOnlyIntegrationParams, gated_memory_read, init_readonly_integration
from scripts.hz0b_b11_baseline_comparison import D_MODEL, FACT_POS, KEY_DIM, SEED, VALUE_DIM, load_frozen_model, make_prompts, targets_for

NUM_SLOTS = 8
WRITE_POS = FACT_POS + 1  # the fact-id token itself -- earliest knowable point, matching B7's WRITE_POS convention


def init_content_proj(seed: int) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    key = mx.random.key(seed + 5000)
    k1, k2 = mx.random.split(key)
    scale = (2.0 / D_MODEL) ** 0.5
    key_proj_w = mx.random.normal((D_MODEL, KEY_DIM), key=k1) * scale
    key_proj_b = mx.zeros((KEY_DIM,))
    value_proj_w = mx.random.normal((D_MODEL, VALUE_DIM), key=k2) * scale
    value_proj_b = mx.zeros((VALUE_DIM,))
    return key_proj_w, key_proj_b, value_proj_w, value_proj_b


def run_oracle_timing_learned_content(model, train_tokens, train_is_a, held_out_tokens, held_out_is_a, *, seed: int, steps: int, lr: float) -> float:
    read_params = init_readonly_integration(D_MODEL, KEY_DIM, VALUE_DIM, seed=seed)
    key_proj_w, key_proj_b, value_proj_w, value_proj_b = init_content_proj(seed)
    params_dict = {"query_w": read_params.query_w, "query_b": read_params.query_b, "gate_w": read_params.gate_w,
                   "gate_b": read_params.gate_b, "value_to_hidden_w": read_params.value_to_hidden_w,
                   "value_to_hidden_b": read_params.value_to_hidden_b, "key_proj_w": key_proj_w, "key_proj_b": key_proj_b,
                   "value_proj_w": value_proj_w, "value_proj_b": value_proj_b}

    train_hidden, _ = frozen_hidden_states(model, train_tokens)
    held_out_hidden, _ = frozen_hidden_states(model, held_out_tokens)
    train_targets = targets_for(train_is_a)
    held_out_targets = targets_for(held_out_is_a)
    train_batch = train_tokens.shape[0]
    held_out_batch = held_out_tokens.shape[0]

    def forward(pd: dict, hidden: mx.array, batch: int) -> mx.array:
        write_hidden = hidden[:, WRITE_POS, :]
        key = write_hidden @ pd["key_proj_w"] + pd["key_proj_b"]
        value = write_hidden @ pd["value_proj_w"] + pd["value_proj_b"]
        memory_state = MemoryState(
            keys=mx.zeros((batch, NUM_SLOTS, KEY_DIM)), values=mx.zeros((batch, NUM_SLOTS, VALUE_DIM)),
            confidence=mx.zeros((batch, NUM_SLOTS)), age=mx.zeros((batch, NUM_SLOTS), dtype=mx.int32),
            protection=mx.zeros((batch, NUM_SLOTS)), write_count=mx.zeros((batch, NUM_SLOTS), dtype=mx.int32),
            last_write_step=mx.zeros((batch, NUM_SLOTS), dtype=mx.int32), write_source=mx.zeros((batch, NUM_SLOTS), dtype=mx.int32),
        )
        memory_state, _, _ = memory_write(memory_state, key, value, mx.ones((batch,)), step=WRITE_POS, slot_idx=mx.zeros((batch,), dtype=mx.int32))
        rp = ReadOnlyIntegrationParams(query_w=pd["query_w"], query_b=pd["query_b"], gate_w=pd["gate_w"],
                                        gate_b=pd["gate_b"], value_to_hidden_w=pd["value_to_hidden_w"], value_to_hidden_b=pd["value_to_hidden_b"])
        output, _ = gated_memory_read(rp, hidden[:, -1, :], memory_state)
        return logits_from_hidden(model, output[:, None, :])[:, -1, :]

    def loss_fn(pd: dict) -> mx.array:
        logits = forward(pd, train_hidden, train_batch)
        return mx.mean(nn.losses.cross_entropy(logits, train_targets))

    grad_fn = mx.value_and_grad(loss_fn)
    for step in range(steps):
        loss, grads = grad_fn(params_dict)
        mx.eval(loss)
        params_dict = {k: params_dict[k] - lr * grads[k] for k in params_dict}
        mx.eval(*params_dict.values())
        if step % 300 == 0 or step == steps - 1:
            print(f"    [oracle-timing-learned-content seed={seed}] step {step:4d}  train loss {float(loss):.5f}")

    logits = forward(params_dict, held_out_hidden, held_out_batch)
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
    print(f"train_count={args.train_count} held_out_count={args.held_out_count} write_pos={WRITE_POS}")

    print(f"\nOracle timing, learned content, oracle slot -- tests content encoder ({args.num_seeds} seeds):")
    accs = []
    for seed_offset in range(args.num_seeds):
        acc = run_oracle_timing_learned_content(model, train_tokens, train_is_a, held_out_tokens, held_out_is_a, seed=SEED + seed_offset, steps=args.steps, lr=args.lr)
        print(f"  seed {SEED + seed_offset}: {acc:.3f}")
        accs.append(acc)
    mean = sum(accs) / len(accs)
    std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
    print(f"\n--- Summary ---\nmean: {mean:.3f}  std: {std:.3f}  range: {min(accs):.3f}-{max(accs):.3f}")
    print("\nReference points:")
    print("  floor (no memory):                   0.000")
    print("  cell 1, oracle-everything:            0.306")
    print("  cell 3, learned timing only:          0.053")
    print("  soft-gate full HZ-0B (all learned):   0.191")
    print("  STE full HZ-0B (all learned):         0.269")
    print("  equal-param adapter (no memory):      0.512")


if __name__ == "__main__":
    main()
