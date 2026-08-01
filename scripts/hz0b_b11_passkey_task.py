"""HZ-0B B11: passkey retrieval task -- one of the plan's 16 named eval
tasks, not yet covered by any prior B11 script (those were 2-way fact
discrimination; this is genuine N-way exact-value retrieval, the
classic "passkey" evaluation format).

Same real frozen HZ-0A checkpoint, same task SHAPE as
scripts/hz0b_b11_baseline_comparison.py (marker token, delayed-recall
gap, read trigger), generalized from a 2-way to a PASSKEY_COUNT-way
choice -- the passkey identity token appears inline in the context
(same "fair test" reasoning as the earlier task: a no-memory model has
a real, non-structural chance to solve it via attention). Starts
directly from the KNOWN-GOOD `lambda_sparse=0.1` fix
(docs/restart/hz0b_b11_evaluation_results.md, "The culminating test")
rather than rediscovering the sparsity-penalty bug on a new task.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from reference.hz0a_mlx_model import HZ0AMlxModel
from reference.hz0b_b6_hz0a_integration import frozen_hidden_states
from reference.hz0b_b8_latent_write import LatentWriteControllerParams, forward as latent_forward_pass, init_latent_write_controller
from reference.hz0b_b11_equal_param_adapter import adapter_forward, init_equal_param_adapter, param_count
from reference.hz0b_b11_equal_param_adapter import forward as adapter_forward_pass
from reference.hz0b_readonly_integration import ReadOnlyIntegrationParams
from reference.hz0b_write_integration import WriteControllerParams

VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF = 24576, 768, 31, 12, 2304
ATTENTION_INDICES = (4, 9, 14, 19, 24, 29)
CHECKPOINT = Path("outputs/hz0a_stage2_100m_hybrid_seed7/native_metal_checkpoint_best_full_holdout")

PASSKEY_COUNT = 4
PASSKEY_MARKER = 22000
PASSKEY_IDS = [22001 + i for i in range(PASSKEY_COUNT)]
READ_TRIGGER = 22010
TARGETS = [22011 + i for i in range(PASSKEY_COUNT)]
FACT_POS = 6
MIDDLE_LEN = 24
PROMPT_LEN = FACT_POS + 2 + MIDDLE_LEN + 1
SEED = 555
ADAPTER_HIDDEN = 450  # matches hz0b_b11_baseline_comparison.py's own 692,418-param budget
KEY_DIM = VALUE_DIM = 32
NUM_SLOTS = 8
LAMBDA_SPARSE = 0.1  # the validated fix, not the original buggy 5.0 -- see docs/restart/hz0b_b11_evaluation_results.md


def load_frozen_model():
    payload = json.loads((CHECKPOINT / "state.json").read_text())
    model = HZ0AMlxModel(VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF, ATTENTION_INDICES, native_metal=True)
    model_arrays = [(item["key"], mx.load(str(CHECKPOINT / item["file"]))) for item in payload["arrays"] if item["group"] == "model"]
    model.update(tree_unflatten(model_arrays))
    mx.eval(model.parameters())
    return model, payload


def make_prompts(count: int, rng: random.Random) -> tuple[mx.array, mx.array]:
    """Returns (token_ids [count, PROMPT_LEN], passkey_index [count])."""
    rows, indices = [], []
    for _ in range(count):
        prefix = [rng.randint(100, VOCAB_SIZE - 100) for _ in range(FACT_POS)]
        idx = rng.randrange(PASSKEY_COUNT)
        middle = [rng.randint(100, VOCAB_SIZE - 100) for _ in range(MIDDLE_LEN)]
        row = prefix + [PASSKEY_MARKER, PASSKEY_IDS[idx]] + middle + [READ_TRIGGER]
        rows.append(row)
        indices.append(idx)
    return mx.array(rows, dtype=mx.int32), mx.array(indices, dtype=mx.int32)


def targets_for(passkey_index: mx.array) -> mx.array:
    targets_table = mx.array(TARGETS)
    return targets_table[passkey_index]


def latent_params_to_dict(p: LatentWriteControllerParams) -> dict:
    d = {f"key_proj.{n}": v for n, v in (("w", p.key_proj_w), ("b", p.key_proj_b))}
    d.update({f"value_proj.{n}": v for n, v in (("w", p.value_proj_w), ("b", p.value_proj_b))})
    d["occupancy_gate_w"] = p.occupancy_gate_w
    wc = p.write_controller
    d.update({f"read_params.{f.name}": getattr(wc.read_params, f.name) for f in dataclasses.fields(wc.read_params)})
    for f in dataclasses.fields(wc):
        if f.name != "read_params":
            d[f"wc.{f.name}"] = getattr(wc, f.name)
    return d


def dict_to_latent_params(d: dict) -> LatentWriteControllerParams:
    read_fields = {k.split(".", 1)[1]: v for k, v in d.items() if k.startswith("read_params.")}
    read_params = ReadOnlyIntegrationParams(**read_fields)
    wc_fields = {k.split(".", 1)[1]: v for k, v in d.items() if k.startswith("wc.")}
    write_controller = WriteControllerParams(read_params=read_params, **wc_fields)
    return LatentWriteControllerParams(
        write_controller=write_controller,
        key_proj_w=d["key_proj.w"], key_proj_b=d["key_proj.b"],
        value_proj_w=d["value_proj.w"], value_proj_b=d["value_proj.b"],
        occupancy_gate_w=d["occupancy_gate_w"],
    )


def run_true_floor(model, held_out_hidden, held_out_idx) -> float:
    logits, _ = adapter_forward_pass(model, precomputed_hidden=held_out_hidden, adapter_params=None)
    predicted = mx.argmax(logits[:, -1, :], axis=-1)
    targets = targets_for(held_out_idx)
    return float(mx.mean((predicted == targets).astype(mx.float32)))


def run_equal_param_adapter(model, train_hidden, train_idx, held_out_hidden, held_out_idx, *, seed: int, steps: int, lr: float) -> float:
    """`train_hidden`/`held_out_hidden` are PRECOMPUTED frozen-backbone
    hidden states (2026-08-01 caching optimization -- see
    reference/hz0b_b11_equal_param_adapter.py::forward's docstring)."""
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


def run_hzb_memory(model, train_hidden, train_idx, held_out_hidden, held_out_idx, *, seed: int, steps: int, lr: float) -> float:
    """`train_hidden`/`held_out_hidden` are PRECOMPUTED frozen-backbone
    hidden states -- see reference/hz0b_b8_latent_write.py::forward's
    docstring."""
    init_params = init_latent_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=seed)
    params_dict = latent_params_to_dict(init_params)
    targets = targets_for(train_idx)

    def loss_fn(pd: dict) -> mx.array:
        p = dict_to_latent_params(pd)
        logits, _, gates = latent_forward_pass(model, precomputed_hidden=train_hidden, latent_params=p, num_slots=NUM_SLOTS)
        task_loss = mx.mean(nn.losses.cross_entropy(logits[:, -1, :], targets))
        return task_loss + LAMBDA_SPARSE * mx.mean(gates)

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
    return float(mx.mean((predicted == targets_for(held_out_idx)).astype(mx.float32)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--held-out-count", type=int, default=80)
    args = parser.parse_args()

    print(f"passkey_count={PASSKEY_COUNT} (chance={1/PASSKEY_COUNT:.3f}) adapter budget={param_count(D_MODEL, ADAPTER_HIDDEN)} memory budget=692,837")
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")

    rng = random.Random(SEED)
    train_tokens, train_idx = make_prompts(args.train_count, rng)
    held_out_tokens, held_out_idx = make_prompts(args.held_out_count, rng)
    print(f"train_count={args.train_count} held_out_count={args.held_out_count} lambda_sparse={LAMBDA_SPARSE}")

    # 2026-08-01 caching optimization -- see
    # scripts/hz0b_b11_baseline_comparison.py's own copy of this comment.
    train_hidden, _ = frozen_hidden_states(model, train_tokens)
    held_out_hidden, _ = frozen_hidden_states(model, held_out_tokens)
    mx.eval(train_hidden, held_out_hidden)

    floor_acc = run_true_floor(model, held_out_hidden, held_out_idx)
    print(f"\n1. True floor (chance={1/PASSKEY_COUNT:.3f}): {floor_acc:.3f}")

    print(f"\n2. Equal-parameter no-memory adapter ({args.num_seeds} seeds):")
    adapter_accs = []
    for i in range(args.num_seeds):
        acc = run_equal_param_adapter(model, train_hidden, train_idx, held_out_hidden, held_out_idx, seed=SEED + i, steps=args.steps, lr=args.lr)
        print(f"  seed {SEED + i}: {acc:.3f}")
        adapter_accs.append(acc)
    adapter_mean = sum(adapter_accs) / len(adapter_accs)
    adapter_std = (sum((a - adapter_mean) ** 2 for a in adapter_accs) / len(adapter_accs)) ** 0.5
    print(f"  mean: {adapter_mean:.3f}  std: {adapter_std:.3f}  range: {min(adapter_accs):.3f}-{max(adapter_accs):.3f}")

    print(f"\n3. HZ-0B real memory, lambda_sparse={LAMBDA_SPARSE} ({args.num_seeds} seeds):")
    memory_accs = []
    for i in range(args.num_seeds):
        acc = run_hzb_memory(model, train_hidden, train_idx, held_out_hidden, held_out_idx, seed=SEED + i, steps=args.steps, lr=args.lr)
        print(f"  seed {SEED + i}: {acc:.3f}")
        memory_accs.append(acc)
    memory_mean = sum(memory_accs) / len(memory_accs)
    memory_std = (sum((a - memory_mean) ** 2 for a in memory_accs) / len(memory_accs)) ** 0.5
    print(f"  mean: {memory_mean:.3f}  std: {memory_std:.3f}  range: {min(memory_accs):.3f}-{max(memory_accs):.3f}")

    print(f"\n--- Summary (passkey task, {PASSKEY_COUNT}-way, chance={1/PASSKEY_COUNT:.3f}) ---")
    print(f"floor:  {floor_acc:.3f}")
    print(f"adapter: mean {adapter_mean:.3f} std {adapter_std:.3f}")
    print(f"memory:  mean {memory_mean:.3f} std {memory_std:.3f}")


if __name__ == "__main__":
    main()
