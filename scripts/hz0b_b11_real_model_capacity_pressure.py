"""HZ-0B B11: real-model capacity pressure. Closes the last of the
disclosed gaps named repeatedly this session
(`docs/restart/hz0b_b11_stage5_baseline_results.md`,
`docs/restart/hz0b_b8_stage5_results.md`, `hz0b_b11_real_model_distractor_immunity_results.md`):
B8 Stage 5's own capacity-pressure scenario is pure-simulator (facts
written via direct MemoryState.write() calls, oracle slot assignment,
no LM, no learned write timing/slot choice). This tests the REAL,
LEARNED write mechanism under genuine capacity pressure: 10 distinct
facts are presented in a single example but the memory controller has
only 8 slots (`NUM_SLOTS=8` < `NUM_FACTS_PER_EXAMPLE=10`) -- unlike
every earlier B11 real-model task (which always wrote far fewer facts
per example than available slots), this forces real within-example
competition for capacity, exercised through the LEARNED write gate,
not an oracle round-robin.

Same real frozen HZ-0A checkpoint, `precomputed_hidden` caching, the
validated `lambda_sparse=0.1` + `target_write_rate=0.1` config. A
single random query per example asks for one of the 10 facts' values.
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

NUM_FACTS_PER_EXAMPLE = 10
ENTITY_MARKERS = [20100 + i for i in range(NUM_FACTS_PER_EXAMPLE)]  # one distinct marker per fact "slot position"
VALUE_MARKER = 20200
NUM_VALUES = 4
VALUE_IDS = [20210 + i for i in range(NUM_VALUES)]
QUERY_MARKER = 20220
TARGETS = [20230 + i for i in range(NUM_VALUES)]
READ_TRIGGER = 20240
assert TARGETS[-1] < VOCAB_SIZE and READ_TRIGGER < VOCAB_SIZE
FACT_POS = 6
PAD_LEN = 3
PROMPT_LEN = FACT_POS + (3 + PAD_LEN) * NUM_FACTS_PER_EXAMPLE + 2 + PAD_LEN + 1
ADAPTER_HIDDEN = 450
KEY_DIM = VALUE_DIM = 32
NUM_SLOTS = 8  # < NUM_FACTS_PER_EXAMPLE=10 -- real capacity pressure
LAMBDA_SPARSE = 0.1
TARGET_WRITE_RATE = 0.1
SEED = 555


def load_frozen_model():
    payload = json.loads((CHECKPOINT / "state.json").read_text())
    model = HZ0AMlxModel(VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF, ATTENTION_INDICES, native_metal=True)
    model_arrays = [(item["key"], mx.load(str(CHECKPOINT / item["file"]))) for item in payload["arrays"] if item["group"] == "model"]
    model.update(tree_unflatten(model_arrays))
    mx.eval(model.parameters())
    return model, payload


def make_prompts(count: int, rng: random.Random) -> tuple[mx.array, mx.array, mx.array]:
    """Returns (token_ids [count, PROMPT_LEN], target_value_idx [count], query_position [count])."""
    rows, target_idx, query_pos = [], [], []
    for _ in range(count):
        row = [rng.randint(100, VOCAB_SIZE - 100) for _ in range(FACT_POS)]
        values_this_example = [rng.randrange(NUM_VALUES) for _ in range(NUM_FACTS_PER_EXAMPLE)]
        for k in range(NUM_FACTS_PER_EXAMPLE):
            row += [ENTITY_MARKERS[k], VALUE_MARKER, VALUE_IDS[values_this_example[k]]]
            row += [rng.randint(100, VOCAB_SIZE - 100) for _ in range(PAD_LEN)]
        q = rng.randrange(NUM_FACTS_PER_EXAMPLE)
        row += [QUERY_MARKER, ENTITY_MARKERS[q]]
        row += [rng.randint(100, VOCAB_SIZE - 100) for _ in range(PAD_LEN)]
        row += [READ_TRIGGER]
        rows.append(row)
        target_idx.append(values_this_example[q])
        query_pos.append(q)
    return mx.array(rows, dtype=mx.int32), mx.array(target_idx, dtype=mx.int32), mx.array(query_pos, dtype=mx.int32)


def targets_for(target_idx: mx.array) -> mx.array:
    targets_table = mx.array(TARGETS)
    return targets_table[target_idx]


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


def run_hzb_memory(model, train_hidden, train_idx, held_out_hidden, held_out_idx, *, seed: int, steps: int, lr: float):
    init_params = init_latent_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=seed)
    params_dict = latent_params_to_dict(init_params)
    targets = targets_for(train_idx)

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
    correct = (predicted == targets_for(held_out_idx))
    return float(mx.mean(correct.astype(mx.float32))), correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--held-out-count", type=int, default=80)
    args = parser.parse_args()

    print(f"num_facts_per_example={NUM_FACTS_PER_EXAMPLE} num_slots={NUM_SLOTS} (capacity pressure: facts > slots) PROMPT_LEN={PROMPT_LEN}")
    print(f"adapter budget={param_count(D_MODEL, ADAPTER_HIDDEN)} memory budget=692,837")
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")

    rng = random.Random(SEED)
    train_tokens, train_idx, _ = make_prompts(args.train_count, rng)
    held_out_tokens, held_out_idx, held_out_query_pos = make_prompts(args.held_out_count, rng)
    print(f"train_count={args.train_count} held_out_count={args.held_out_count} lambda_sparse={LAMBDA_SPARSE} target_write_rate={TARGET_WRITE_RATE}")

    train_hidden, _ = frozen_hidden_states(model, train_tokens)
    held_out_hidden, _ = frozen_hidden_states(model, held_out_tokens)
    mx.eval(train_hidden, held_out_hidden)

    floor_acc = run_true_floor(model, held_out_hidden, held_out_idx)
    print(f"\n1. True floor (chance=0.250): {floor_acc:.3f}")

    print(f"\n2. Equal-parameter no-memory adapter ({args.num_seeds} seeds):")
    adapter_accs = []
    for i in range(args.num_seeds):
        acc = run_equal_param_adapter(model, train_hidden, train_idx, held_out_hidden, held_out_idx, seed=SEED + i, steps=args.steps, lr=args.lr)
        print(f"  seed {SEED + i}: {acc:.3f}")
        adapter_accs.append(acc)
    adapter_mean = sum(adapter_accs) / len(adapter_accs)
    adapter_std = (sum((a - adapter_mean) ** 2 for a in adapter_accs) / len(adapter_accs)) ** 0.5
    print(f"  mean: {adapter_mean:.3f}  std: {adapter_std:.3f}  range: {min(adapter_accs):.3f}-{max(adapter_accs):.3f}")

    print(f"\n3. HZ-0B real memory ({args.num_seeds} seeds):")
    memory_accs = []
    last_correct = None
    for i in range(args.num_seeds):
        acc, correct = run_hzb_memory(model, train_hidden, train_idx, held_out_hidden, held_out_idx, seed=SEED + i, steps=args.steps, lr=args.lr)
        print(f"  seed {SEED + i}: {acc:.3f}")
        memory_accs.append(acc)
        last_correct = correct
    memory_mean = sum(memory_accs) / len(memory_accs)
    memory_std = (sum((a - memory_mean) ** 2 for a in memory_accs) / len(memory_accs)) ** 0.5
    print(f"  mean: {memory_mean:.3f}  std: {memory_std:.3f}  range: {min(memory_accs):.3f}-{max(memory_accs):.3f}")

    print(f"\n4. Retrieval accuracy by query position (last seed only, {NUM_FACTS_PER_EXAMPLE} fact positions, recency check):")
    for pos in range(NUM_FACTS_PER_EXAMPLE):
        mask = (held_out_query_pos == pos)
        n = int(mx.sum(mask))
        if n > 0:
            pos_acc = float(mx.sum(last_correct.astype(mx.float32) * mask.astype(mx.float32)) / n)
            print(f"  position {pos:2d} (n={n:2d}): {pos_acc:.3f}")

    print(f"\n--- Summary (real-model capacity pressure, {NUM_FACTS_PER_EXAMPLE} facts / {NUM_SLOTS} slots) ---")
    print(f"floor:  {floor_acc:.3f}")
    print(f"adapter: mean {adapter_mean:.3f} std {adapter_std:.3f}")
    print(f"memory:  mean {memory_mean:.3f} std {memory_std:.3f}")


if __name__ == "__main__":
    main()
