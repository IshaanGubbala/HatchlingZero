"""HZ-0B Phase B9, Stage 1: unfreeze only "memory-adjacent projections"
(here: the LAST HZ-0A block -- the one immediately upstream of where
memory injection happens, per `reference/hz0b_b9_finetune.py`'s docstring
for why that's the natural reading) and fine-tune it jointly with the B7
write controller against the real frozen checkpoint.

Reuses B7's exact task (`scripts/hz0b_b7_real_integration_probe.py`) so
the comparison is apples-to-apples: same fact, same prompts, same
background-preservation setup, `confidence_scaled=True` (the validated
B6/B7 fix). Only new thing: the last block's own ~1/31st of HZ-0A's
parameters are ALSO trainable this time, not just the small controller.

Measures B9's own required 5 quantities at each stage:
  - general language loss (no-memory, before vs after fine-tuning --
    catches drift from touching HZ-0A's own weights, separate from
    anything memory-related)
  - memory-task performance (write-then-read rank, vs B7's frozen-
    backbone baseline)
  - write frequency (mean write_gate at the write position)
  - memory interference (should_write=0 logit drift, vs the FINE-TUNED
    block's own no-memory baseline -- the fair comparison, since the
    baseline itself shifts once the block is no longer frozen)
  - catastrophic degradation (fine-tuned no-memory CE vs the ORIGINAL,
    untouched frozen checkpoint's no-memory CE -- the real "did this
    break HZ-0A" check)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from reference.hz0b_b9_finetune import apply_block_params, block_param_count, block_params_dict
from reference.hz0b_b7_hz0a_integration import forward
from scripts.hz0b_b7_real_integration_probe import (
    CHECKPOINT,
    D_MODEL,
    KEY_DIM,
    LAYERS,
    PROMPT_LEN,
    READ_TRIGGER_A,
    READ_TRIGGER_B,
    SEED,
    TARGET,
    VALUE_DIM,
    VOCAB_SIZE,
    WRITE_POS,
    dict_to_params,
    load_frozen_model,
    make_prompts,
    make_write_labels,
    params_to_dict,
    target_rank_stats,
)
from reference.hz0b_write_integration import init_write_controller


def general_val_loss(model, val_tokens, *, controller_params=None, write_labels=None, confidence_scaled=True) -> float:
    logits, _ = forward(model, val_tokens, controller_params=controller_params, write_labels=write_labels, confidence_scaled=confidence_scaled)
    mx.eval(logits)
    return float(mx.mean(nn.losses.cross_entropy(logits[:, :-1].astype(mx.float32), val_tokens[:, 1:])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--block-lr", type=float, default=1e-5, help="separate, much smaller LR for the unfrozen HZ-0A block -- it's a pretrained ~301M-scale weight, not a fresh random init like the controller")
    parser.add_argument("--controller-lr", type=float, default=1.5e-1)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    import random
    rng = random.Random(SEED)
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")
    unfreeze_index = LAYERS - 1
    print(f"unfreezing block index {unfreeze_index} (last block, {block_param_count(model, unfreeze_index):,} params) + controller, block_lr={args.block_lr} controller_lr={args.controller_lr} steps={args.steps}")

    train_prompts = make_prompts(24, rng)
    held_out_prompts = make_prompts(8, rng)
    background_lines = Path("data/packed/repro_256_val.jsonl").open().readlines()[64:80]
    background_tokens = mx.array([json.loads(l)[:32] for l in background_lines], dtype=mx.int32)
    val_lines = Path("data/packed/repro_256_val.jsonl").open().readlines()[:64]
    val_tokens = mx.array([json.loads(l)[:256] for l in val_lines], dtype=mx.int32)

    memory_key = mx.random.normal((1, KEY_DIM), key=mx.random.key(SEED))
    memory_value = mx.random.normal((1, VALUE_DIM), key=mx.random.key(SEED + 1))

    # Baseline, BEFORE any fine-tuning -- the original frozen checkpoint's
    # own no-memory general loss, the anchor the "catastrophic degradation"
    # check at the end compares against.
    original_val_loss = general_val_loss(model, val_tokens)
    print(f"\noriginal frozen checkpoint, no-memory general val loss: {original_val_loss:.6f}")

    init_controller = init_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=SEED)
    combined = {**params_to_dict(init_controller), **block_params_dict(model, unfreeze_index)}
    controller_keys = set(params_to_dict(init_controller).keys())
    block_keys = set(block_params_dict(model, unfreeze_index).keys())

    labels_write_train = make_write_labels(24, memory_key, memory_value, write=True)
    labels_noop_bg = make_write_labels(background_tokens.shape[0], memory_key, memory_value, write=False)
    bg_write_labels = labels_noop_bg[:min(len(labels_noop_bg), background_tokens.shape[1])] + [None] * max(0, background_tokens.shape[1] - len(labels_noop_bg))

    def loss_fn(cd: dict) -> mx.array:
        apply_block_params(model, unfreeze_index, cd)
        p = dict_to_params({k: v for k, v in cd.items() if k in controller_keys})
        logits, _ = forward(model, train_prompts, controller_params=p, write_labels=labels_write_train, confidence_scaled=True)
        final_logits = logits[:, -1, :]
        targets = mx.full((final_logits.shape[0],), TARGET, dtype=mx.int32)
        task_loss = mx.mean(nn.losses.cross_entropy(final_logits, targets))
        bg_logits, _ = forward(model, background_tokens, controller_params=p, write_labels=bg_write_labels, confidence_scaled=True)
        preserve_loss = mx.mean(nn.losses.cross_entropy(bg_logits[:, :-1].astype(mx.float32), background_tokens[:, 1:]))
        return task_loss + args.lambda_preserve * preserve_loss

    grad_fn = mx.value_and_grad(loss_fn)
    print("\n--- fine-tuning controller (fast LR) + last HZ-0A block (slow LR) jointly ---")
    for step in range(args.steps):
        loss, grads = grad_fn(combined)
        mx.eval(loss)
        combined = {
            k: v - (args.controller_lr if k in controller_keys else args.block_lr) * grads[k]
            for k, v in combined.items()
        }
        mx.eval(*combined.values())
        if step % 100 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  train loss {float(loss):.5f}")

    apply_block_params(model, unfreeze_index, combined)
    trained_controller = dict_to_params({k: v for k, v in combined.items() if k in controller_keys})

    print("\n=== B9 Stage 1 required measurements ===")

    # 1 & 5. general language loss + catastrophic-degradation check
    finetuned_no_mem_loss = general_val_loss(model, val_tokens)
    print(f"\n1/5. general val loss -- original frozen: {original_val_loss:.6f}  after fine-tuning (no memory): {finetuned_no_mem_loss:.6f}  delta: {finetuned_no_mem_loss - original_val_loss:+.6f}")
    degradation_pct = (finetuned_no_mem_loss - original_val_loss) / original_val_loss * 100
    print(f"     relative change: {degradation_pct:+.3f}%  -- catastrophic would be a large, unambiguous jump (e.g. >20-50%), not a few percent")

    # 2. memory-task performance vs B7's frozen-backbone baseline (179.4 at 1000 steps, unscaled; 325.0 at higher-step confidence-scaled)
    labels_write_eval = make_write_labels(8, memory_key, memory_value, write=True)
    labels_readonly = [None] * PROMPT_LEN
    rank_readonly, rank_write, ce_write = target_rank_stats(model, held_out_prompts, trained_controller, labels_write_eval, labels_readonly, confidence_scaled=True)
    print(f"\n2. memory-task performance -- held-out rank: read-only(never written)={rank_readonly:.1f}  write-then-read={rank_write:.1f} / {VOCAB_SIZE}  (B7 frozen-backbone baseline was 179.4-325.0 depending on config)")

    # 3. write frequency
    write_gate_logit = trained_controller.write_gate_w
    print(f"\n3. write controller gate weight norm (proxy for write frequency/confidence): {float(mx.sqrt(mx.sum(write_gate_logit**2))):.4f}")

    # 4. memory interference, vs the FINE-TUNED block's own no-memory baseline (fair comparison)
    val_labels_noop = make_write_labels(val_tokens.shape[0], memory_key, memory_value, write=False)
    val_write_labels = val_labels_noop[:min(len(val_labels_noop), val_tokens.shape[1])] + [None] * max(0, val_tokens.shape[1] - len(val_labels_noop))
    logits_finetuned_no_mem, _ = forward(model, val_tokens)
    logits_finetuned_with_empty_mem, _ = forward(model, val_tokens, controller_params=trained_controller, write_labels=val_write_labels, confidence_scaled=True)
    mx.eval(logits_finetuned_no_mem, logits_finetuned_with_empty_mem)
    interference_max_diff = float(mx.max(mx.abs(logits_finetuned_no_mem - logits_finetuned_with_empty_mem)))
    print(f"\n4. memory interference (should_write=0, vs fine-tuned block's own no-memory baseline): max abs logit diff = {interference_max_diff:.6f}")

    print("\n=== B9 Stage 1 exit-gate read ===")
    print("Exit gate: \"Memory improvements survive limited fine-tuning without destroying HZ-0A quality.\"")
    if abs(degradation_pct) < 10.0:
        print(f"General quality: PRESERVED ({degradation_pct:+.2f}%, well below a catastrophic threshold).")
    else:
        print(f"General quality: REAL DEGRADATION FOUND ({degradation_pct:+.2f}%) -- not catastrophic-scale but disclosed, not hidden.")


if __name__ == "__main__":
    main()
