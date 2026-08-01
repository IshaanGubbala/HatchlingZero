"""HZ-0B B11 continuation: write-mechanism diagnosis, factorial cell 3.

| Write timing | Write content | Slot choice | Interpretation |
| --- | --- | --- | --- |
| Oracle | Oracle | Oracle | Tests read path and memory substrate (run, see hz0b_b11_write_diagnosis_oracle_all.py: mean 0.306) |
| Oracle | Learned | Oracle | Tests content encoder (not run yet) |
| **Learned** | **Oracle** | **Oracle** | **Tests write-trigger policy (this script)** |
| Oracle | Oracle | Learned | Tests addressing (not run yet) |
| Learned | Learned | Learned | Full HZ-0B (already run: soft 0.191, STE 0.269) |

Cell 1 (oracle-everything) already showed the read path itself is a real
bottleneck (mean 0.306, below the 0.512 adapter). This cell asks: does
ALSO making the model learn WHEN to write (content and slot still
handed to it for free) make things measurably worse than cell 1's 0.306
-- and if so, by how much, relative to the full learned mechanism's
0.191-0.269?

Every position gets a chance to write (no `should_write` label, matching
B8 Stage 3's own latent framing), gated by a single learned linear+sigmoid
head trained jointly with the read path and a write-sparsity penalty.
Content is the SAME fixed, distinguishable oracle key/value pair cell 1
used (constant per batch row, based on which fact that example actually
shows) -- the model never has to encode WHAT to write, only decide
WHETHER a given position is a good moment to commit the (already-known)
correct write.
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
from reference.hz0b_write_integration import _blend_state_by_row
from scripts.hz0b_b11_baseline_comparison import D_MODEL, KEY_DIM, SEED, VALUE_DIM, load_frozen_model, make_prompts, targets_for
from scripts.hz0b_b11_write_diagnosis_oracle_all import oracle_memory_state

LAMBDA_SPARSE = 5.0
NUM_SLOTS = 8


def init_gate(seed: int) -> tuple[mx.array, mx.array]:
    key = mx.random.key(seed + 4000)
    scale = (2.0 / D_MODEL) ** 0.5
    return mx.random.normal((D_MODEL, 1), key=key) * scale, mx.zeros((1,))


def sequential_learned_timing_oracle_content(read_params: ReadOnlyIntegrationParams, gate_w: mx.array, gate_b: mx.array, hidden: mx.array, oracle_key_per_row: mx.array, oracle_value_per_row: mx.array) -> tuple[mx.array, mx.array]:
    """hidden: [batch, seq, d_model]. oracle_key/value_per_row: [batch,
    key_dim]/[batch, value_dim] -- the SAME oracle content offered at
    every position (content is not position-dependent; only WHETHER to
    write it is learned). Returns (output_hidden, write_gates [batch, seq])."""
    batch, seq, _ = hidden.shape
    memory_state = MemoryState(
        keys=mx.zeros((batch, NUM_SLOTS, KEY_DIM)), values=mx.zeros((batch, NUM_SLOTS, VALUE_DIM)),
        confidence=mx.zeros((batch, NUM_SLOTS)), age=mx.zeros((batch, NUM_SLOTS), dtype=mx.int32),
        protection=mx.zeros((batch, NUM_SLOTS)), write_count=mx.zeros((batch, NUM_SLOTS), dtype=mx.int32),
        last_write_step=mx.zeros((batch, NUM_SLOTS), dtype=mx.int32), write_source=mx.zeros((batch, NUM_SLOTS), dtype=mx.int32),
    )
    outputs, gates = [], []
    for t in range(seq):
        h_t = hidden[:, t, :]
        output, _ = gated_memory_read(read_params, h_t, memory_state)
        write_gate = mx.sigmoid((h_t @ gate_w + gate_b)[:, 0])
        candidate_state, _, _ = memory_write(memory_state, oracle_key_per_row, oracle_value_per_row, write_gate, step=t)
        memory_state = _blend_state_by_row(memory_state, candidate_state, write_gate)
        outputs.append(output)
        gates.append(write_gate)
    return mx.stack(outputs, axis=1), mx.stack(gates, axis=1)


def run_learned_timing(model, train_tokens, train_is_a, held_out_tokens, held_out_is_a, *, seed: int, steps: int, lr: float, lambda_sparse: float = LAMBDA_SPARSE) -> float:
    read_params = init_readonly_integration(D_MODEL, KEY_DIM, VALUE_DIM, seed=seed)
    gate_w, gate_b = init_gate(seed)
    params_dict = {"query_w": read_params.query_w, "query_b": read_params.query_b, "gate_w_read": read_params.gate_w,
                   "gate_b_read": read_params.gate_b, "value_to_hidden_w": read_params.value_to_hidden_w,
                   "value_to_hidden_b": read_params.value_to_hidden_b, "write_gate_w": gate_w, "write_gate_b": gate_b}

    train_hidden, _ = frozen_hidden_states(model, train_tokens)
    held_out_hidden, _ = frozen_hidden_states(model, held_out_tokens)
    train_targets = targets_for(train_is_a)
    held_out_targets = targets_for(held_out_is_a)

    # Oracle key/value per row -- reuse cell 1's exact fixed constants via
    # its 1-slot oracle state, then broadcast to every position here.
    train_oracle = oracle_memory_state(train_is_a, KEY_DIM, VALUE_DIM)
    held_out_oracle = oracle_memory_state(held_out_is_a, KEY_DIM, VALUE_DIM)
    train_key, train_value = train_oracle.keys[:, 0, :], train_oracle.values[:, 0, :]
    held_out_key, held_out_value = held_out_oracle.keys[:, 0, :], held_out_oracle.values[:, 0, :]

    def loss_fn(pd: dict) -> mx.array:
        rp = ReadOnlyIntegrationParams(query_w=pd["query_w"], query_b=pd["query_b"], gate_w=pd["gate_w_read"],
                                        gate_b=pd["gate_b_read"], value_to_hidden_w=pd["value_to_hidden_w"], value_to_hidden_b=pd["value_to_hidden_b"])
        output, gates = sequential_learned_timing_oracle_content(rp, pd["write_gate_w"], pd["write_gate_b"], train_hidden, train_key, train_value)
        logits = logits_from_hidden(model, output)[:, -1, :]
        task_loss = mx.mean(nn.losses.cross_entropy(logits, train_targets))
        return task_loss + lambda_sparse * mx.mean(gates)

    grad_fn = mx.value_and_grad(loss_fn)
    for step in range(steps):
        loss, grads = grad_fn(params_dict)
        mx.eval(loss)
        params_dict = {k: params_dict[k] - lr * grads[k] for k in params_dict}
        mx.eval(*params_dict.values())
        if step % 300 == 0 or step == steps - 1:
            print(f"    [learned-timing seed={seed}] step {step:4d}  train loss {float(loss):.5f}")

    rp = ReadOnlyIntegrationParams(query_w=params_dict["query_w"], query_b=params_dict["query_b"], gate_w=params_dict["gate_w_read"],
                                    gate_b=params_dict["gate_b_read"], value_to_hidden_w=params_dict["value_to_hidden_w"], value_to_hidden_b=params_dict["value_to_hidden_b"])
    output, _ = sequential_learned_timing_oracle_content(rp, params_dict["write_gate_w"], params_dict["write_gate_b"], held_out_hidden, held_out_key, held_out_value)
    logits = logits_from_hidden(model, output)[:, -1, :]
    predicted = mx.argmax(logits, axis=-1)
    return float(mx.mean((predicted == held_out_targets).astype(mx.float32)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--held-out-count", type=int, default=64)
    parser.add_argument("--lambda-sparse", type=float, default=LAMBDA_SPARSE, help="tests whether the sparsity penalty is what starves the timing gate's gradient signal (docs/restart/hz0b_b11_evaluation_results.md's causal hypothesis for cell 3's collapse)")
    args = parser.parse_args()

    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")

    rng = random.Random(SEED)
    train_tokens, train_is_a = make_prompts(args.train_count, rng)
    held_out_tokens, held_out_is_a = make_prompts(args.held_out_count, rng)
    print(f"train_count={args.train_count} held_out_count={args.held_out_count} lambda_sparse={args.lambda_sparse}")

    print(f"\nLearned timing, oracle content, oracle slot -- tests write-trigger policy ({args.num_seeds} seeds):")
    accs = []
    for seed_offset in range(args.num_seeds):
        acc = run_learned_timing(model, train_tokens, train_is_a, held_out_tokens, held_out_is_a, seed=SEED + seed_offset, steps=args.steps, lr=args.lr, lambda_sparse=args.lambda_sparse)
        print(f"  seed {SEED + seed_offset}: {acc:.3f}")
        accs.append(acc)
    mean = sum(accs) / len(accs)
    std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
    print(f"\n--- Summary ---\nmean: {mean:.3f}  std: {std:.3f}  range: {min(accs):.3f}-{max(accs):.3f}")
    print("\nReference points:")
    print("  floor (no memory):                   0.000")
    print("  oracle-everything (cell 1):           0.306")
    print("  soft-gate full HZ-0B (all learned):   0.191")
    print("  STE full HZ-0B (all learned):         0.269")
    print("  equal-param adapter (no memory):      0.512")
    if mean >= 0.28:
        print("\nRESULT: learning WHEN to write, alone, does not meaningfully hurt vs. cell 1's oracle-everything -- "
              "the write-trigger policy is not the dominant bottleneck.")
    else:
        print("\nRESULT: learning WHEN to write, alone, measurably hurts vs. cell 1's oracle-everything -- "
              "the write-trigger policy IS a real, independent contributor to the failure.")


if __name__ == "__main__":
    main()
