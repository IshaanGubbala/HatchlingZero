"""HZ-0B Phase B9, Stage 2: unfreeze SEVERAL "selected upper HZ-0A
layers" (the last N blocks, not just one) and fine-tune jointly with the
B7 write controller. Builds directly on Stage 1's corrected result
(`docs/restart/hz0b_b9_stage1_results.md`): unfreezing the last block,
after a controller-only warmup, gave a real (if modest) improvement over
frozen-only. This asks whether unfreezing MORE of the upper stack helps
further, using the same warmup-then-joint recipe.

Multi-seed from the start this time (`--seeds`, default 3) -- Stage 1's
first pass didn't do this and drew a wrong conclusion from a single run;
applying that lesson up front here rather than retrofitting it after a
mistake.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from reference.hz0b_b9_finetune import apply_multi_block_params, multi_block_param_count, multi_block_params_dict
from reference.hz0b_b7_hz0a_integration import forward
from scripts.hz0b_b7_real_integration_probe import (
    CHECKPOINT,
    D_MODEL,
    KEY_DIM,
    LAYERS,
    PROMPT_LEN,
    TARGET,
    VALUE_DIM,
    VOCAB_SIZE,
    dict_to_params,
    load_frozen_model,
    make_prompts,
    make_write_labels,
    params_to_dict,
    target_rank_stats,
)
from reference.hz0b_write_integration import init_write_controller


def general_val_loss(model, val_tokens) -> float:
    logits, _ = forward(model, val_tokens)
    mx.eval(logits)
    return float(mx.mean(nn.losses.cross_entropy(logits[:, :-1].astype(mx.float32), val_tokens[:, 1:])))


def run_one_seed(seed: int, block_indices: list[int], warmup_steps: int, joint_steps: int, controller_lr: float, block_lr: float, lambda_preserve: float) -> dict:
    rng = random.Random(seed)
    model, _ = load_frozen_model()

    train_prompts = make_prompts(24, rng)
    held_out_prompts = make_prompts(8, rng)
    background_lines = Path("data/packed/repro_256_val.jsonl").open().readlines()[64:80]
    background_tokens = mx.array([json.loads(l)[:32] for l in background_lines], dtype=mx.int32)
    val_lines = Path("data/packed/repro_256_val.jsonl").open().readlines()[:64]
    val_tokens = mx.array([json.loads(l)[:256] for l in val_lines], dtype=mx.int32)

    memory_key = mx.random.normal((1, KEY_DIM), key=mx.random.key(seed))
    memory_value = mx.random.normal((1, VALUE_DIM), key=mx.random.key(seed + 1))

    original_val_loss = general_val_loss(model, val_tokens)

    init_controller = init_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=seed)
    combined = {**params_to_dict(init_controller), **multi_block_params_dict(model, block_indices)}
    controller_keys = set(params_to_dict(init_controller).keys())

    labels_write_train = make_write_labels(24, memory_key, memory_value, write=True)
    labels_noop_bg = make_write_labels(background_tokens.shape[0], memory_key, memory_value, write=False)
    bg_write_labels = labels_noop_bg[:min(len(labels_noop_bg), background_tokens.shape[1])] + [None] * max(0, background_tokens.shape[1] - len(labels_noop_bg))

    def loss_fn(cd: dict) -> mx.array:
        apply_multi_block_params(model, block_indices, cd)
        p = dict_to_params({k: v for k, v in cd.items() if k in controller_keys})
        logits, _ = forward(model, train_prompts, controller_params=p, write_labels=labels_write_train, confidence_scaled=True)
        final_logits = logits[:, -1, :]
        targets = mx.full((final_logits.shape[0],), TARGET, dtype=mx.int32)
        task_loss = mx.mean(nn.losses.cross_entropy(final_logits, targets))
        bg_logits, _ = forward(model, background_tokens, controller_params=p, write_labels=bg_write_labels, confidence_scaled=True)
        preserve_loss = mx.mean(nn.losses.cross_entropy(bg_logits[:, :-1].astype(mx.float32), background_tokens[:, 1:]))
        return task_loss + lambda_preserve * preserve_loss

    grad_fn = mx.value_and_grad(loss_fn)

    for step in range(warmup_steps):
        loss, grads = grad_fn(combined)
        mx.eval(loss)
        combined = {k: (v - controller_lr * grads[k] if k in controller_keys else v) for k, v in combined.items()}
        mx.eval(*combined.values())

    for step in range(joint_steps):
        loss, grads = grad_fn(combined)
        mx.eval(loss)
        combined = {k: v - (controller_lr if k in controller_keys else block_lr) * grads[k] for k, v in combined.items()}
        mx.eval(*combined.values())

    apply_multi_block_params(model, block_indices, combined)
    trained_controller = dict_to_params({k: v for k, v in combined.items() if k in controller_keys})

    labels_write_eval = make_write_labels(8, memory_key, memory_value, write=True)
    labels_readonly = [None] * PROMPT_LEN
    _, rank_write, _ = target_rank_stats(model, held_out_prompts, trained_controller, labels_write_eval, labels_readonly, confidence_scaled=True)

    finetuned_val_loss = general_val_loss(model, val_tokens)

    return {
        "seed": seed, "final_train_loss": float(loss), "rank_write": rank_write,
        "original_val_loss": original_val_loss, "finetuned_val_loss": finetuned_val_loss,
        "val_loss_delta_pct": (finetuned_val_loss - original_val_loss) / original_val_loss * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-upper-layers", type=int, default=3, help="unfreeze the LAST N blocks (Stage 1 used N=1)")
    parser.add_argument("--warmup-steps", type=int, default=2500)
    parser.add_argument("--joint-steps", type=int, default=1000)
    parser.add_argument("--controller-lr", type=float, default=0.4)
    parser.add_argument("--block-lr", type=float, default=1e-5)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[321, 322, 323])
    args = parser.parse_args()

    block_indices = list(range(LAYERS - args.num_upper_layers, LAYERS))
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")
    print(f"unfreezing blocks {block_indices} ({multi_block_param_count(model, block_indices):,} params) + controller")
    print(f"warmup_steps={args.warmup_steps} joint_steps={args.joint_steps} controller_lr={args.controller_lr} block_lr={args.block_lr} seeds={args.seeds}")

    results = []
    for seed in args.seeds:
        r = run_one_seed(seed, block_indices, args.warmup_steps, args.joint_steps, args.controller_lr, args.block_lr, args.lambda_preserve)
        results.append(r)
        print(f"seed={seed}  final_train_loss={r['final_train_loss']:.5f}  rank_write={r['rank_write']:.1f}  val_loss_delta={r['val_loss_delta_pct']:+.3f}%")

    ranks = [r["rank_write"] for r in results]
    deltas = [r["val_loss_delta_pct"] for r in results]
    print("\n=== summary across seeds ===")
    print(f"rank_write: mean={statistics.mean(ranks):.1f}  stdev={statistics.pstdev(ranks):.1f}  values={[round(v,1) for v in ranks]}")
    print(f"val_loss_delta_pct: mean={statistics.mean(deltas):+.3f}%  stdev={statistics.pstdev(deltas):.3f}%  values={[round(v,3) for v in deltas]}")


if __name__ == "__main__":
    main()
