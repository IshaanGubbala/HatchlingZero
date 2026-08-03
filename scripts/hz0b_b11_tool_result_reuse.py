"""HZ-0B B11: tool-result reuse -- the last of the plan's 16 named
eval tasks. Deliberately NOT a re-skin of the earlier recall tasks
(baseline_comparison, passkey, long-conversation): those all test
"can the correct value be regurgitated later." This tests REUSE -- a
"tool call" returns an ordinal result value; later, a threshold is
presented; the correct answer depends on comparing the stored result
against the threshold (`GREATER` vs `NOT_GREATER`), which requires
actually using the stored value in a downstream decision, not just
retrieving it verbatim.

Same real frozen HZ-0A checkpoint, `precomputed_hidden` caching, and
the validated `lambda_sparse=0.1` + `target_write_rate=0.1` config.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
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

TOOL_MARKER = 20000
NUM_LEVELS = 4
RESULT_IDS = [20010 + i for i in range(NUM_LEVELS)]
COMPARE_MARKER = 20020
THRESHOLD_IDS = [20030 + i for i in range(NUM_LEVELS)]
READ_TRIGGER = 20040
GREATER_TARGET, NOT_GREATER_TARGET = 20050, 20051
assert NOT_GREATER_TARGET < VOCAB_SIZE, "all special token ids must be < VOCAB_SIZE or they are unreachable as predictions"
FACT_POS = 6
PAD_LEN = 10
PROMPT_LEN = FACT_POS + 2 + PAD_LEN + 2 + PAD_LEN + 1
ADAPTER_HIDDEN = 450
KEY_DIM = VALUE_DIM = 32
NUM_SLOTS = 8
LAMBDA_SPARSE = 0.1
TARGET_WRITE_RATE = 0.1
LAMBDA_READ_ENTROPY = 0.01
LAMBDA_VALUE_PRESERVE = 0.01
SEED = 555


def load_frozen_model():
    payload = json.loads((CHECKPOINT / "state.json").read_text())
    model = HZ0AMlxModel(VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF, ATTENTION_INDICES, native_metal=True)
    model_arrays = [(item["key"], mx.load(str(CHECKPOINT / item["file"]))) for item in payload["arrays"] if item["group"] == "model"]
    model.update(tree_unflatten(model_arrays))
    mx.eval(model.parameters())
    return model, payload


def make_prompts(count: int, rng: random.Random) -> tuple[mx.array, mx.array]:
    """Returns (token_ids [count, PROMPT_LEN], is_greater [count])."""
    rows, is_greater = [], []
    # Cycle through every result/threshold combination so the controller sees
    # the comparison rule itself, not a sparse accidental subset of pairs.
    combinations = [(result_idx, threshold_idx) for result_idx in range(NUM_LEVELS) for threshold_idx in range(NUM_LEVELS)]
    for example_idx in range(count):
        row = [rng.randint(100, VOCAB_SIZE - 100) for _ in range(FACT_POS)]
        result_idx, threshold_idx = combinations[example_idx % len(combinations)]
        row += [TOOL_MARKER, RESULT_IDS[result_idx]]
        row += [rng.randint(100, VOCAB_SIZE - 100) for _ in range(PAD_LEN)]
        row += [COMPARE_MARKER, THRESHOLD_IDS[threshold_idx]]
        row += [rng.randint(100, VOCAB_SIZE - 100) for _ in range(PAD_LEN)]
        row += [READ_TRIGGER]
        rows.append(row)
        is_greater.append(1.0 if result_idx > threshold_idx else 0.0)
    return mx.array(rows, dtype=mx.int32), mx.array(is_greater)


def targets_for(is_greater: mx.array) -> mx.array:
    return mx.where(is_greater > 0.5, mx.array(GREATER_TARGET), mx.array(NOT_GREATER_TARGET)).astype(mx.int32)


def run_true_floor(model, held_out_hidden, held_out_is_greater) -> float:
    logits, _ = adapter_forward_pass(model, precomputed_hidden=held_out_hidden, adapter_params=None)
    predicted = mx.argmax(logits[:, -1, :], axis=-1)
    targets = targets_for(held_out_is_greater)
    return float(mx.mean((predicted == targets).astype(mx.float32)))


def _apply_update(params_dict: dict, grads: dict, *, optimizer, optimizer_name: str, lr: float) -> dict:
    if optimizer_name == "adam":
        return optimizer.apply_gradients(grads, params_dict)
    return {k: params_dict[k] - lr * grads[k] for k in params_dict}


def run_equal_param_adapter(model, train_hidden, train_is_greater, held_out_hidden, held_out_is_greater, *, seed: int, steps: int, lr: float, optimizer_name: str) -> float:
    params = init_equal_param_adapter(D_MODEL, ADAPTER_HIDDEN, seed=seed)
    params_dict = {"w1": params.w1, "b1": params.b1, "w2": params.w2, "b2": params.b2}
    targets = targets_for(train_is_greater)
    optimizer = optim.Adam(learning_rate=lr) if optimizer_name == "adam" else None

    def loss_fn(pd: dict) -> mx.array:
        p = type(params)(**pd)
        logits, _ = adapter_forward_pass(model, precomputed_hidden=train_hidden, adapter_params=p)
        return mx.mean(nn.losses.cross_entropy(logits[:, -1, :], targets))

    grad_fn = mx.value_and_grad(loss_fn)
    for step in range(steps):
        loss, grads = grad_fn(params_dict)
        mx.eval(loss)
        params_dict = _apply_update(params_dict, grads, optimizer=optimizer, optimizer_name=optimizer_name, lr=lr)
        mx.eval(*params_dict.values())
        if step % 300 == 0 or step == steps - 1:
            print(f"    [adapter seed={seed}] step {step:4d}  train loss {float(loss):.5f}")

    trained = type(params)(**params_dict)
    logits, _ = adapter_forward_pass(model, precomputed_hidden=held_out_hidden, adapter_params=trained)
    predicted = mx.argmax(logits[:, -1, :], axis=-1)
    return float(mx.mean((predicted == targets_for(held_out_is_greater)).astype(mx.float32)))


def run_hzb_memory(model, train_hidden, train_is_greater, held_out_hidden, held_out_is_greater, *, seed: int, steps: int, lr: float, optimizer_name: str) -> float:
    init_params = init_latent_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=seed)
    params_dict = latent_params_to_dict(init_params)
    targets = targets_for(train_is_greater)
    optimizer = optim.Adam(learning_rate=lr) if optimizer_name == "adam" else None

    def loss_fn(pd: dict) -> mx.array:
        p = dict_to_latent_params(pd)
        logits, _, gates, read_entropy = latent_forward_pass(model, precomputed_hidden=train_hidden, latent_params=p, num_slots=NUM_SLOTS, read_hops=1, return_read_entropy=True)
        task_loss = mx.mean(nn.losses.cross_entropy(logits[:, -1, :], targets))
        write_rate = mx.mean(gates)
        sparsity_loss = (write_rate - TARGET_WRITE_RATE) ** 2
        value = train_hidden @ p.value_proj_w + p.value_proj_b
        reconstructed = value @ p.write_controller.read_params.value_to_hidden_w + p.write_controller.read_params.value_to_hidden_b
        preserve_loss = mx.mean((reconstructed - train_hidden) ** 2) / (mx.mean(train_hidden ** 2) + 1e-6)
        return task_loss + LAMBDA_SPARSE * sparsity_loss + LAMBDA_READ_ENTROPY * mx.mean(read_entropy) + LAMBDA_VALUE_PRESERVE * preserve_loss

    grad_fn = mx.value_and_grad(loss_fn)
    for step in range(steps):
        loss, grads = grad_fn(params_dict)
        mx.eval(loss)
        params_dict = _apply_update(params_dict, grads, optimizer=optimizer, optimizer_name=optimizer_name, lr=lr)
        mx.eval(*params_dict.values())
        if step % 300 == 0 or step == steps - 1:
            print(f"    [memory seed={seed}] step {step:4d}  train loss {float(loss):.5f}")

    trained = dict_to_latent_params(params_dict)
    logits, _, _ = latent_forward_pass(model, precomputed_hidden=held_out_hidden, latent_params=trained, num_slots=NUM_SLOTS, read_hops=1)
    predicted = mx.argmax(logits[:, -1, :], axis=-1)
    return float(mx.mean((predicted == targets_for(held_out_is_greater)).astype(mx.float32)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=SEED)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--train-count", type=int, default=320, help="Balanced-scale training set; 80 examples under-cover result/threshold combinations")
    parser.add_argument("--held-out-count", type=int, default=80)
    args = parser.parse_args()

    print(f"num_levels={NUM_LEVELS} PROMPT_LEN={PROMPT_LEN}")
    print(f"adapter budget={param_count(D_MODEL, ADAPTER_HIDDEN)} memory budget=692,837")
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")

    rng = random.Random(SEED)
    train_tokens, train_is_greater = make_prompts(args.train_count, rng)
    held_out_tokens, held_out_is_greater = make_prompts(args.held_out_count, rng)
    base_rate = float(mx.mean(held_out_is_greater))
    print(f"train_count={args.train_count} held_out_count={args.held_out_count} held_out_greater_rate={base_rate:.3f} lambda_sparse={LAMBDA_SPARSE} target_write_rate={TARGET_WRITE_RATE} optimizer={args.optimizer}")

    train_hidden, _ = frozen_hidden_states(model, train_tokens)
    held_out_hidden, _ = frozen_hidden_states(model, held_out_tokens)
    mx.eval(train_hidden, held_out_hidden)

    floor_acc = run_true_floor(model, held_out_hidden, held_out_is_greater)
    print(f"\n1. True floor (majority-class baseline={max(base_rate, 1-base_rate):.3f}): {floor_acc:.3f}")

    print(f"\n2. Equal-parameter no-memory adapter ({args.num_seeds} seeds):")
    adapter_accs = []
    for i in range(args.num_seeds):
        acc = run_equal_param_adapter(model, train_hidden, train_is_greater, held_out_hidden, held_out_is_greater, seed=args.seed_start + i, steps=args.steps, lr=args.lr, optimizer_name=args.optimizer)
        print(f"  seed {args.seed_start + i}: {acc:.3f}")
        adapter_accs.append(acc)
    adapter_mean = sum(adapter_accs) / len(adapter_accs)
    adapter_std = (sum((a - adapter_mean) ** 2 for a in adapter_accs) / len(adapter_accs)) ** 0.5
    print(f"  mean: {adapter_mean:.3f}  std: {adapter_std:.3f}  range: {min(adapter_accs):.3f}-{max(adapter_accs):.3f}")

    print(f"\n3. HZ-0B real memory ({args.num_seeds} seeds):")
    memory_accs = []
    for i in range(args.num_seeds):
        acc = run_hzb_memory(model, train_hidden, train_is_greater, held_out_hidden, held_out_is_greater, seed=args.seed_start + i, steps=args.steps, lr=args.lr, optimizer_name=args.optimizer)
        print(f"  seed {args.seed_start + i}: {acc:.3f}")
        memory_accs.append(acc)
    memory_mean = sum(memory_accs) / len(memory_accs)
    memory_std = (sum((a - memory_mean) ** 2 for a in memory_accs) / len(memory_accs)) ** 0.5
    print(f"  mean: {memory_mean:.3f}  std: {memory_std:.3f}  range: {min(memory_accs):.3f}-{max(memory_accs):.3f}")

    print(f"\n--- Summary (tool-result reuse: stored result vs later threshold) ---")
    print(f"floor:  {floor_acc:.3f}")
    print(f"adapter: mean {adapter_mean:.3f} std {adapter_std:.3f}")
    print(f"memory:  mean {memory_mean:.3f} std {memory_std:.3f}")


if __name__ == "__main__":
    main()
