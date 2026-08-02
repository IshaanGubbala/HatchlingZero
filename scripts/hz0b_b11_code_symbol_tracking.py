"""HZ-0B B11: code-symbol tracking -- one of the plan's 16 named eval
tasks, not yet covered. Distinct SHAPE from the earlier fact-recall
tasks (baseline_comparison, passkey): here the SAME symbol is
reassigned multiple times (3 sequential writes to one key, each
separated by padding so proximity-to-read-trigger can't trivially
solve it), and the correct answer is whichever value was assigned
LAST -- a real test of "overwrite tracking" (must discard stale
writes to the same key), not just "recall a single written fact".
Mirrors B8 Stage 4's "variable reassignment" item type, but real-model
+ the real learned write/read mechanism instead of pure natural-
sequence composition.

Same real frozen HZ-0A checkpoint, same `precomputed_hidden` caching,
same validated `target_write_rate=0.1` config
(docs/restart/hz0b_b11_evaluation_results.md) rather than
rediscovering the sparsity-penalty issue on a new task.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from reference.hz0a_mlx_model import HZ0AMlxModel
from reference.hz0b_b6_hz0a_integration import frozen_hidden_states
from reference.hz0b_b8_latent_write import forward as latent_forward_pass, init_latent_write_controller
from reference.hz0b_b11_equal_param_adapter import init_equal_param_adapter, param_count
from reference.hz0b_b11_equal_param_adapter import forward as adapter_forward_pass
from scripts.hz0b_b11_passkey_task import latent_params_to_dict, dict_to_latent_params

VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF = 24576, 768, 31, 12, 2304
ATTENTION_INDICES = (4, 9, 14, 19, 24, 29)
CHECKPOINT = Path("outputs/hz0a_stage2_100m_hybrid_seed7/native_metal_checkpoint_best_full_holdout")

SYMBOL_MARKER, SYMBOL_ID = 23000, 23001
ASSIGN_MARKER = 23002
NUM_VALUES = 4
VALUE_IDS = [23010 + i for i in range(NUM_VALUES)]
TARGETS = [23020 + i for i in range(NUM_VALUES)]
READ_TRIGGER = 23030
NUM_REASSIGNMENTS = 3
FACT_POS = 6
PAD_LEN = 8
PROMPT_LEN = FACT_POS + 2 + (PAD_LEN + 2) * NUM_REASSIGNMENTS + PAD_LEN + 1
ADAPTER_HIDDEN = 450
KEY_DIM = VALUE_DIM = 32
NUM_SLOTS = 8
LAMBDA_SPARSE = 0.1
TARGET_WRITE_RATE = 0.1  # 2026-08-01 validated fix, see hz0b_b11_evaluation_results.md
SEED = 555


def load_frozen_model():
    payload = json.loads((CHECKPOINT / "state.json").read_text())
    model = HZ0AMlxModel(VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF, ATTENTION_INDICES, native_metal=True)
    model_arrays = [(item["key"], mx.load(str(CHECKPOINT / item["file"]))) for item in payload["arrays"] if item["group"] == "model"]
    model.update(tree_unflatten(model_arrays))
    mx.eval(model.parameters())
    return model, payload


def make_prompts(count: int, rng: random.Random) -> tuple[mx.array, mx.array]:
    """Returns (token_ids [count, PROMPT_LEN], final_value_index [count])."""
    rows, final_idx = [], []
    for _ in range(count):
        row = [rng.randint(100, VOCAB_SIZE - 100) for _ in range(FACT_POS)]
        row += [SYMBOL_MARKER, SYMBOL_ID]
        last = None
        for _ in range(NUM_REASSIGNMENTS):
            row += [rng.randint(100, VOCAB_SIZE - 100) for _ in range(PAD_LEN)]
            idx = rng.randrange(NUM_VALUES)
            row += [ASSIGN_MARKER, VALUE_IDS[idx]]
            last = idx
        row += [rng.randint(100, VOCAB_SIZE - 100) for _ in range(PAD_LEN)]
        row += [READ_TRIGGER]
        rows.append(row)
        final_idx.append(last)
    return mx.array(rows, dtype=mx.int32), mx.array(final_idx, dtype=mx.int32)


def targets_for(final_idx: mx.array) -> mx.array:
    targets_table = mx.array(TARGETS)
    return targets_table[final_idx]


def run_true_floor(model, held_out_hidden, held_out_idx) -> float:
    logits, _ = adapter_forward_pass(model, precomputed_hidden=held_out_hidden, adapter_params=None)
    predicted = mx.argmax(logits[:, -1, :], axis=-1)
    targets = targets_for(held_out_idx)
    return float(mx.mean((predicted == targets).astype(mx.float32)))


def run_equal_param_adapter(model, train_hidden, train_idx, held_out_hidden, held_out_idx, *, seed: int, steps: int, lr: float) -> float:
    params = init_equal_param_adapter(D_MODEL, ADAPTER_HIDDEN, seed=seed)
    params_dict = {"w1": params.w1, "b1": params.b1, "w2": params.w2, "b2": params.b2}
    targets = targets_for(train_idx)

    def loss_fn(pd: dict) -> mx.array:
        p = type(params)(**pd)
        logits, _ = adapter_forward_pass(model, precomputed_hidden=train_hidden, adapter_params=p)
        return mx.mean(nn.losses.cross_entropy(logits[:, -1, :], targets))

    grad_fn = mx.value_and_grad(loss_fn)
    for step in range(steps):
        loss, grads = grad_fn(params_dict)
        mx.eval(loss)
        params_dict = {k: params_dict[k] - lr * grads[k] for k in params_dict}
        mx.eval(*params_dict.values())
        if step % 300 == 0 or step == steps - 1:
            print(f"    [adapter seed={seed}] step {step:4d}  train loss {float(loss):.5f}")

    trained = type(params)(**params_dict)
    logits, _ = adapter_forward_pass(model, precomputed_hidden=held_out_hidden, adapter_params=trained)
    predicted = mx.argmax(logits[:, -1, :], axis=-1)
    return float(mx.mean((predicted == targets_for(held_out_idx)).astype(mx.float32)))


def run_hzb_memory(model, train_hidden, train_idx, held_out_hidden, held_out_idx, *, seed: int, steps: int, lr: float, ste: bool = False) -> float:
    """`ste` (2026-08-01): hard/discrete write decisions via a straight-
    through estimator instead of the continuous blend -- tests whether
    this fixes the read-focus failure diagnosed in
    docs/restart/hz0b_b11_write_slot_diagnosis_code_symbol_results.md
    (write routing is clean, but the READ step fails to reliably
    retrieve the correctly-written slot; the leading hypothesis is
    diluted-by-filler-position writes, which STE's all-or-nothing
    commits should reduce)."""
    init_params = init_latent_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=seed)
    params_dict = latent_params_to_dict(init_params)
    targets = targets_for(train_idx)

    def loss_fn(pd: dict) -> mx.array:
        p = dict_to_latent_params(pd)
        logits, _, gates = latent_forward_pass(model, precomputed_hidden=train_hidden, latent_params=p, num_slots=NUM_SLOTS, ste=ste)
        task_loss = mx.mean(nn.losses.cross_entropy(logits[:, -1, :], targets))
        write_rate = mx.mean(gates)
        sparsity_loss = (write_rate - TARGET_WRITE_RATE) ** 2
        return task_loss + LAMBDA_SPARSE * sparsity_loss

    grad_fn = mx.value_and_grad(loss_fn)
    for step in range(steps):
        loss, grads = grad_fn(params_dict)
        mx.eval(loss)
        params_dict = {k: params_dict[k] - lr * grads[k] for k in params_dict}
        mx.eval(*params_dict.values())
        if step % 300 == 0 or step == steps - 1:
            print(f"    [memory seed={seed} ste={ste}] step {step:4d}  train loss {float(loss):.5f}")

    trained = dict_to_latent_params(params_dict)
    logits, _, _ = latent_forward_pass(model, precomputed_hidden=held_out_hidden, latent_params=trained, num_slots=NUM_SLOTS, ste=ste)
    predicted = mx.argmax(logits[:, -1, :], axis=-1)
    return float(mx.mean((predicted == targets_for(held_out_idx)).astype(mx.float32)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--held-out-count", type=int, default=80)
    parser.add_argument("--ste", action="store_true", help="hard/discrete write decisions via STE -- tests the fix candidate named in the read-focus root-cause diagnosis")
    args = parser.parse_args()

    print(f"num_reassignments={NUM_REASSIGNMENTS} num_values={NUM_VALUES} (chance={1/NUM_VALUES:.3f}) adapter budget={param_count(D_MODEL, ADAPTER_HIDDEN)} memory budget=692,837")
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")

    rng = random.Random(SEED)
    train_tokens, train_idx = make_prompts(args.train_count, rng)
    held_out_tokens, held_out_idx = make_prompts(args.held_out_count, rng)
    print(f"train_count={args.train_count} held_out_count={args.held_out_count} lambda_sparse={LAMBDA_SPARSE} target_write_rate={TARGET_WRITE_RATE}")

    train_hidden, _ = frozen_hidden_states(model, train_tokens)
    held_out_hidden, _ = frozen_hidden_states(model, held_out_tokens)
    mx.eval(train_hidden, held_out_hidden)

    floor_acc = run_true_floor(model, held_out_hidden, held_out_idx)
    print(f"\n1. True floor (chance={1/NUM_VALUES:.3f}): {floor_acc:.3f}")

    print(f"\n2. Equal-parameter no-memory adapter ({args.num_seeds} seeds):")
    adapter_accs = []
    for i in range(args.num_seeds):
        acc = run_equal_param_adapter(model, train_hidden, train_idx, held_out_hidden, held_out_idx, seed=SEED + i, steps=args.steps, lr=args.lr)
        print(f"  seed {SEED + i}: {acc:.3f}")
        adapter_accs.append(acc)
    adapter_mean = sum(adapter_accs) / len(adapter_accs)
    adapter_std = (sum((a - adapter_mean) ** 2 for a in adapter_accs) / len(adapter_accs)) ** 0.5
    print(f"  mean: {adapter_mean:.3f}  std: {adapter_std:.3f}  range: {min(adapter_accs):.3f}-{max(adapter_accs):.3f}")

    print(f"\n3. HZ-0B real memory, lambda_sparse={LAMBDA_SPARSE}, target_write_rate={TARGET_WRITE_RATE}, ste={args.ste} ({args.num_seeds} seeds):")
    memory_accs = []
    for i in range(args.num_seeds):
        acc = run_hzb_memory(model, train_hidden, train_idx, held_out_hidden, held_out_idx, seed=SEED + i, steps=args.steps, lr=args.lr, ste=args.ste)
        print(f"  seed {SEED + i}: {acc:.3f}")
        memory_accs.append(acc)
    memory_mean = sum(memory_accs) / len(memory_accs)
    memory_std = (sum((a - memory_mean) ** 2 for a in memory_accs) / len(memory_accs)) ** 0.5
    print(f"  mean: {memory_mean:.3f}  std: {memory_std:.3f}  range: {min(memory_accs):.3f}-{max(memory_accs):.3f}")

    print(f"\n--- Summary (code-symbol tracking, {NUM_REASSIGNMENTS} reassignments, {NUM_VALUES}-way, chance={1/NUM_VALUES:.3f}) ---")
    print(f"floor:  {floor_acc:.3f}")
    print(f"adapter: mean {adapter_mean:.3f} std {adapter_std:.3f}")
    print(f"memory:  mean {memory_mean:.3f} std {memory_std:.3f}")


if __name__ == "__main__":
    main()
