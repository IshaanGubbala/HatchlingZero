"""HZ-0B B11: long-conversation consistency -- one of the plan's 16
named eval tasks, not yet covered. Same 2-way fact-discrimination task
as scripts/hz0b_b11_baseline_comparison.py (already validated: mean
0.819-0.830 at MIDDLE_LEN=24), but with the gap between the written
fact and the read trigger increased from 24 to 200 tokens -- roughly
an 8x longer "conversation" before the fact needs to be recalled,
while staying comfortably within the 256-token sequence length the
HZ-0A backbone was actually trained at (avoiding out-of-distribution
extrapolation as a separate confound).

Tests whether the validated `lambda_sparse=0.1` +
`target_write_rate=0.1` config's advantage over the equal-param
adapter holds up over a much longer real gap, or degrades toward the
adapter (or toward chance) as the intervening context grows.
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

FACT_MARKER, FACT_A_ID, FACT_B_ID = 21000, 21001, 21002
READ_TRIGGER_A, READ_TRIGGER_B = 21003, 21004
TARGET_A, TARGET_B = 21005, 21006
FACT_POS = 6
MIDDLE_LEN = 200  # 8x the validated baseline's 24-token gap, still within the 256-token trained sequence length
PROMPT_LEN = FACT_POS + 2 + MIDDLE_LEN + 2
SEED = 555
ADAPTER_HIDDEN = 450
KEY_DIM = VALUE_DIM = 32
NUM_SLOTS = 8
LAMBDA_SPARSE = 0.1
TARGET_WRITE_RATE = 0.1


def load_frozen_model():
    payload = json.loads((CHECKPOINT / "state.json").read_text())
    model = HZ0AMlxModel(VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF, ATTENTION_INDICES, native_metal=True)
    model_arrays = [(item["key"], mx.load(str(CHECKPOINT / item["file"]))) for item in payload["arrays"] if item["group"] == "model"]
    model.update(tree_unflatten(model_arrays))
    mx.eval(model.parameters())
    return model, payload


def make_prompts(count: int, rng: random.Random) -> tuple[mx.array, mx.array]:
    rows, fact_is_a = [], []
    for _ in range(count):
        prefix = [rng.randint(100, VOCAB_SIZE - 100) for _ in range(FACT_POS)]
        is_a = rng.random() < 0.5
        fact_id = FACT_A_ID if is_a else FACT_B_ID
        middle = [rng.randint(100, VOCAB_SIZE - 100) for _ in range(MIDDLE_LEN)]
        row = prefix + [FACT_MARKER, fact_id] + middle + [READ_TRIGGER_A, READ_TRIGGER_B]
        rows.append(row)
        fact_is_a.append(1.0 if is_a else 0.0)
    return mx.array(rows, dtype=mx.int32), mx.array(fact_is_a)


def targets_for(is_a: mx.array) -> mx.array:
    return mx.where(is_a > 0.5, mx.array(TARGET_A), mx.array(TARGET_B)).astype(mx.int32)


def run_true_floor(model, held_out_hidden, held_out_is_a) -> float:
    logits, _ = adapter_forward_pass(model, precomputed_hidden=held_out_hidden, adapter_params=None)
    predicted = mx.argmax(logits[:, -1, :], axis=-1)
    targets = targets_for(held_out_is_a)
    return float(mx.mean((predicted == targets).astype(mx.float32)))


def run_equal_param_adapter(model, train_hidden, train_is_a, held_out_hidden, held_out_is_a, *, seed: int, steps: int, lr: float) -> float:
    params = init_equal_param_adapter(D_MODEL, ADAPTER_HIDDEN, seed=seed)
    params_dict = {"w1": params.w1, "b1": params.b1, "w2": params.w2, "b2": params.b2}
    targets = targets_for(train_is_a)

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
    return float(mx.mean((predicted == targets_for(held_out_is_a)).astype(mx.float32)))


def run_hzb_memory(model, train_hidden, train_is_a, held_out_hidden, held_out_is_a, *, seed: int, steps: int, lr: float) -> float:
    init_params = init_latent_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=seed)
    params_dict = latent_params_to_dict(init_params)
    targets = targets_for(train_is_a)

    def loss_fn(pd: dict) -> mx.array:
        p = dict_to_latent_params(pd)
        logits, _, gates = latent_forward_pass(model, precomputed_hidden=train_hidden, latent_params=p, num_slots=NUM_SLOTS)
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
            print(f"    [memory seed={seed}] step {step:4d}  train loss {float(loss):.5f}")

    trained = dict_to_latent_params(params_dict)
    logits, _, _ = latent_forward_pass(model, precomputed_hidden=held_out_hidden, latent_params=trained, num_slots=NUM_SLOTS)
    predicted = mx.argmax(logits[:, -1, :], axis=-1)
    return float(mx.mean((predicted == targets_for(held_out_is_a)).astype(mx.float32)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--held-out-count", type=int, default=64)
    args = parser.parse_args()

    print(f"MIDDLE_LEN={MIDDLE_LEN} (8x the validated 24-token baseline gap) PROMPT_LEN={PROMPT_LEN}")
    print(f"adapter budget={param_count(D_MODEL, ADAPTER_HIDDEN)} memory budget=692,837")
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")

    rng = random.Random(SEED)
    train_tokens, train_is_a = make_prompts(args.train_count, rng)
    held_out_tokens, held_out_is_a = make_prompts(args.held_out_count, rng)
    print(f"train_count={args.train_count} held_out_count={args.held_out_count} lambda_sparse={LAMBDA_SPARSE} target_write_rate={TARGET_WRITE_RATE}")

    train_hidden, _ = frozen_hidden_states(model, train_tokens)
    held_out_hidden, _ = frozen_hidden_states(model, held_out_tokens)
    mx.eval(train_hidden, held_out_hidden)

    floor_acc = run_true_floor(model, held_out_hidden, held_out_is_a)
    print(f"\n1. True floor: {floor_acc:.3f}")

    print(f"\n2. Equal-parameter no-memory adapter ({args.num_seeds} seeds):")
    adapter_accs = []
    for i in range(args.num_seeds):
        acc = run_equal_param_adapter(model, train_hidden, train_is_a, held_out_hidden, held_out_is_a, seed=SEED + i, steps=args.steps, lr=args.lr)
        print(f"  seed {SEED + i}: {acc:.3f}")
        adapter_accs.append(acc)
    adapter_mean = sum(adapter_accs) / len(adapter_accs)
    adapter_std = (sum((a - adapter_mean) ** 2 for a in adapter_accs) / len(adapter_accs)) ** 0.5
    print(f"  mean: {adapter_mean:.3f}  std: {adapter_std:.3f}  range: {min(adapter_accs):.3f}-{max(adapter_accs):.3f}")

    print(f"\n3. HZ-0B real memory ({args.num_seeds} seeds):")
    memory_accs = []
    for i in range(args.num_seeds):
        acc = run_hzb_memory(model, train_hidden, train_is_a, held_out_hidden, held_out_is_a, seed=SEED + i, steps=args.steps, lr=args.lr)
        print(f"  seed {SEED + i}: {acc:.3f}")
        memory_accs.append(acc)
    memory_mean = sum(memory_accs) / len(memory_accs)
    memory_std = (sum((a - memory_mean) ** 2 for a in memory_accs) / len(memory_accs)) ** 0.5
    print(f"  mean: {memory_mean:.3f}  std: {memory_std:.3f}  range: {min(memory_accs):.3f}-{max(memory_accs):.3f}")

    print(f"\n--- Summary (long-conversation consistency, MIDDLE_LEN={MIDDLE_LEN}) ---")
    print(f"floor:  {floor_acc:.3f}")
    print(f"adapter: mean {adapter_mean:.3f} std {adapter_std:.3f}")
    print(f"memory:  mean {memory_mean:.3f} std {memory_std:.3f}")


if __name__ == "__main__":
    main()
