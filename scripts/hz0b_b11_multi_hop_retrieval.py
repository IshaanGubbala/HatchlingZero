"""HZ-0B B11: multi-hop retrieval -- one of the plan's 16 named eval
tasks, not yet covered. Genuinely different SHAPE from the earlier
single-key retrieval tasks (baseline_comparison, passkey): a 2-hop
chain. Hop 1: an entity points to a pointer token (`ENTITY_MARKER,
ENTITY_ID, POINTER_MARKER, pointer_i` -- a within-window association
the frozen backbone's own attention can plausibly resolve on its own,
since it never leaves a short local span). Hop 2: elsewhere in the
sequence, that SAME pointer token is associated with a final value
(`VALUE_MARKER, pointer_i, value_j`) -- genuinely requires carrying
information across a real gap, the part memory is meant to help with.
2 DISTRACTOR value triples (different pointer tokens) are scattered in
too, so the mechanism must match the specific pointer from hop 1, not
just grab the nearest `VALUE_MARKER` pair.

Same real frozen HZ-0A checkpoint, `precomputed_hidden` caching, and
the validated `lambda_sparse=0.1` + `target_write_rate=0.1` config.
"""
from __future__ import annotations

import argparse
import json
import os
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
# Overridable via env vars -- mirrors scripts/hz0b_b11_baseline_comparison.py's
# HZ0_EVAL_CHECKPOINT/HZ0_EVAL_MIXER pattern (see commit 9a2e180) so this
# script can be repointed at a corrected gdn2_fix checkpoint for HZ-0G G2.
CHECKPOINT = Path(os.environ.get(
    "HZ0_EVAL_CHECKPOINT",
    "outputs/hz0a_stage2_100m_hybrid_seed7/native_metal_checkpoint_best_full_holdout",
))
MIXER = os.environ.get("HZ0_EVAL_MIXER", "gdn2")

ENTITY_MARKER, ENTITY_ID = 24000, 24001
POINTER_MARKER = 24002
NUM_POINTERS = 4
POINTER_IDS = [24010 + i for i in range(NUM_POINTERS)]
VALUE_MARKER = 24020
NUM_VALUES = 4
VALUE_IDS = [24030 + i for i in range(NUM_VALUES)]
TARGETS = [24040 + i for i in range(NUM_VALUES)]
READ_TRIGGER = 24050
FACT_POS = 6
PAD_LEN = 6
PROMPT_LEN = FACT_POS + 4 + PAD_LEN + 3 + PAD_LEN + 3 + PAD_LEN + 3 + PAD_LEN + 1
ADAPTER_HIDDEN = 450
KEY_DIM = VALUE_DIM = 32
NUM_SLOTS = 8
DECAY_RATE = 0.99
GRAD_CLIP_NORM = 1.0
LAMBDA_SPARSE = 0.1
TARGET_WRITE_RATE = 0.1
LAMBDA_READ_ENTROPY = 0.01
LAMBDA_VALUE_PRESERVE = 0.001
SEED = 555


def clip_gradients(grads: dict) -> dict:
    norm = mx.sqrt(sum(mx.sum(g * g) for g in grads.values()) + 1e-8)
    scale = mx.minimum(mx.array(1.0), mx.array(GRAD_CLIP_NORM) / norm)
    return {k: g * scale for k, g in grads.items()}


def load_frozen_model(checkpoint: Path = CHECKPOINT, mixer: str = MIXER):
    payload = json.loads((checkpoint / "state.json").read_text())
    model = HZ0AMlxModel(VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF, ATTENTION_INDICES, native_metal=True, mixer=mixer)
    model_arrays = [(item["key"], mx.load(str(checkpoint / item["file"]))) for item in payload["arrays"] if item["group"] == "model"]
    model.update(tree_unflatten(model_arrays))
    mx.eval(model.parameters())
    return model, payload


def make_prompts(count: int, rng: random.Random) -> tuple[mx.array, mx.array]:
    """Returns (token_ids [count, PROMPT_LEN], final_value_index [count])."""
    rows, final_idx = [], []
    for _ in range(count):
        row = [rng.randint(100, VOCAB_SIZE - 100) for _ in range(FACT_POS)]
        pointer_idx = rng.randrange(NUM_POINTERS)
        row += [ENTITY_MARKER, ENTITY_ID, POINTER_MARKER, POINTER_IDS[pointer_idx]]

        distractor_pointer_idxs = [i for i in range(NUM_POINTERS) if i != pointer_idx]
        rng.shuffle(distractor_pointer_idxs)
        d1, d2 = distractor_pointer_idxs[0], distractor_pointer_idxs[1]

        target_value_idx = rng.randrange(NUM_VALUES)
        triples = [
            (d1, rng.randrange(NUM_VALUES)),
            (pointer_idx, target_value_idx),
            (d2, rng.randrange(NUM_VALUES)),
        ]
        rng.shuffle(triples)
        for p_idx, v_idx in triples:
            row += [rng.randint(100, VOCAB_SIZE - 100) for _ in range(PAD_LEN)]
            row += [VALUE_MARKER, POINTER_IDS[p_idx], VALUE_IDS[v_idx]]

        row += [rng.randint(100, VOCAB_SIZE - 100) for _ in range(PAD_LEN)]
        row += [READ_TRIGGER]
        assert len(row) == PROMPT_LEN, (len(row), PROMPT_LEN)
        rows.append(row)
        final_idx.append(target_value_idx)
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


def run_hzb_memory(model, train_hidden, train_idx, held_out_hidden, held_out_idx, *, seed: int, steps: int, lr: float, ste: bool = False, decay_rate: float = DECAY_RATE, shared_key_query: bool = False, read_hops: int = 2, num_slots: int = NUM_SLOTS, target_write_rate: float = TARGET_WRITE_RATE) -> float:
    init_params = init_latent_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=seed)
    params_dict = latent_params_to_dict(init_params)
    targets = targets_for(train_idx)

    def loss_fn(pd: dict) -> mx.array:
        p = dict_to_latent_params(pd)
        logits, _, gates, read_entropy = latent_forward_pass(model, precomputed_hidden=train_hidden, latent_params=p, num_slots=num_slots, read_hops=read_hops, ste=ste, shared_key_query=shared_key_query, decay_rate=decay_rate, return_read_entropy=True)
        task_loss = mx.mean(nn.losses.cross_entropy(logits[:, -1, :], targets))
        write_rate = mx.mean(gates)
        sparsity_loss = (write_rate - target_write_rate) ** 2
        value = train_hidden @ p.value_proj_w + p.value_proj_b
        reconstructed = value @ p.write_controller.read_params.value_to_hidden_w + p.write_controller.read_params.value_to_hidden_b
        preserve_loss = mx.mean((reconstructed - train_hidden) ** 2) / (mx.mean(train_hidden ** 2) + 1e-6)
        return task_loss + LAMBDA_SPARSE * sparsity_loss + LAMBDA_READ_ENTROPY * mx.mean(read_entropy) + LAMBDA_VALUE_PRESERVE * preserve_loss

    grad_fn = mx.value_and_grad(loss_fn)
    for step in range(steps):
        loss, grads = grad_fn(params_dict)
        mx.eval(loss)
        grads = clip_gradients(grads)
        params_dict = {k: params_dict[k] - lr * grads[k] for k in params_dict}
        mx.eval(*params_dict.values())
        if step % 300 == 0 or step == steps - 1:
            print(f"    [memory seed={seed}] step {step:4d}  train loss {float(loss):.5f}")

    trained = dict_to_latent_params(params_dict)
    logits, _, _ = latent_forward_pass(model, precomputed_hidden=held_out_hidden, latent_params=trained, num_slots=num_slots, read_hops=read_hops, ste=ste, shared_key_query=shared_key_query, decay_rate=decay_rate)
    predicted = mx.argmax(logits[:, -1, :], axis=-1)
    return float(mx.mean((predicted == targets_for(held_out_idx)).astype(mx.float32)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=SEED)
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--held-out-count", type=int, default=80)
    parser.add_argument("--ste", action="store_true")
    parser.add_argument("--decay-rate", type=float, default=DECAY_RATE)
    parser.add_argument(
        "--shared-key-query", action=argparse.BooleanOptionalAction, default=True,
        help="use the learned key projection for both hops instead of composing a second query",
    )
    parser.add_argument("--read-hops", type=int, choices=(1, 2, 3), default=2,
                        help="number of sequential memory reads per position")
    parser.add_argument("--num-slots", type=int, choices=(8, 16, 32), default=NUM_SLOTS)
    parser.add_argument("--target-write-rate", type=float, choices=(0.05, 0.1, 0.2, 0.3), default=0.2)
    parser.add_argument("--memory-only", action="store_true", help="skip the matched adapter arm for bounded memory-only sweeps")
    args = parser.parse_args()

    print(f"num_pointers={NUM_POINTERS} num_values={NUM_VALUES} (chance={1/NUM_VALUES:.3f}) PROMPT_LEN={PROMPT_LEN}")
    print(f"adapter budget={param_count(D_MODEL, ADAPTER_HIDDEN)} memory budget=692,837")
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")

    rng = random.Random(SEED)
    train_tokens, train_idx = make_prompts(args.train_count, rng)
    held_out_tokens, held_out_idx = make_prompts(args.held_out_count, rng)
    print(f"train_count={args.train_count} held_out_count={args.held_out_count} lambda_sparse={LAMBDA_SPARSE} target_write_rate={args.target_write_rate} decay_rate={args.decay_rate} shared_key_query={args.shared_key_query} read_hops={args.read_hops} num_slots={args.num_slots}")

    train_hidden, _ = frozen_hidden_states(model, train_tokens)
    held_out_hidden, _ = frozen_hidden_states(model, held_out_tokens)
    mx.eval(train_hidden, held_out_hidden)

    floor_acc = run_true_floor(model, held_out_hidden, held_out_idx)
    print(f"\n1. True floor (chance={1/NUM_VALUES:.3f}): {floor_acc:.3f}")

    adapter_accs = []
    if not args.memory_only:
        print(f"\n2. Equal-parameter no-memory adapter ({args.num_seeds} seeds):")
        for i in range(args.num_seeds):
            acc = run_equal_param_adapter(model, train_hidden, train_idx, held_out_hidden, held_out_idx, seed=args.seed_start + i, steps=args.steps, lr=args.lr)
            print(f"  seed {args.seed_start + i}: {acc:.3f}")
            adapter_accs.append(acc)
        adapter_mean = sum(adapter_accs) / len(adapter_accs)
        adapter_std = (sum((a - adapter_mean) ** 2 for a in adapter_accs) / len(adapter_accs)) ** 0.5
        print(f"  mean: {adapter_mean:.3f}  std: {adapter_std:.3f}  range: {min(adapter_accs):.3f}-{max(adapter_accs):.3f}")

    print(f"\n3. HZ-0B real memory ({args.num_seeds} seeds):")
    memory_accs = []
    for i in range(args.num_seeds):
        acc = run_hzb_memory(model, train_hidden, train_idx, held_out_hidden, held_out_idx, seed=args.seed_start + i, steps=args.steps, lr=args.lr, ste=args.ste, decay_rate=args.decay_rate, shared_key_query=args.shared_key_query, read_hops=args.read_hops, num_slots=args.num_slots, target_write_rate=args.target_write_rate)
        print(f"  seed {args.seed_start + i}: {acc:.3f}")
        memory_accs.append(acc)
    memory_mean = sum(memory_accs) / len(memory_accs)
    memory_std = (sum((a - memory_mean) ** 2 for a in memory_accs) / len(memory_accs)) ** 0.5
    print(f"  mean: {memory_mean:.3f}  std: {memory_std:.3f}  range: {min(memory_accs):.3f}-{max(memory_accs):.3f}")

    print(f"\n--- Summary (multi-hop retrieval, {NUM_VALUES}-way, chance={1/NUM_VALUES:.3f}) ---")
    print(f"floor:  {floor_acc:.3f}")
    if adapter_accs:
        print(f"adapter: mean {adapter_mean:.3f} std {adapter_std:.3f}")
    print(f"memory:  mean {memory_mean:.3f} std {memory_std:.3f}")


if __name__ == "__main__":
    main()
