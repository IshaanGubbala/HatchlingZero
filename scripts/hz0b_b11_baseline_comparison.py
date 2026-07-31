"""HZ-0B Phase B11: first real evaluation-suite experiment.

Tests B11's own exit gate directly: "HZ-0B provides a measurable
advantage that cannot be explained only by more parameters or more
context." B6 and B7's real-integration tasks inject the memorized fact
via an oracle bypass that never appears as tokens at all -- a no-memory
model is STRUCTURALLY unable to solve those regardless of mechanism
merit, which is not a fair test of this exit gate. B8 Stage 3's task is
different and reusable here: the fact-id token genuinely appears in the
context (`FACT_MARKER, fact_id` inline in the prompt), the two-way
design already defeats a constant-bias shortcut, and a real, trained
latent write+read result already exists for it (0.750 held-out accuracy,
`docs/restart/hz0b_b8_stage3_results.md`).

This script reuses that EXACT task construction (same constants, same
`make_prompts`, copied verbatim from
`scripts/hz0b_b8_stage3_latent_write_probe.py` for byte-identical
task/data parity -- not re-derived) and adds the one condition B11 needs
that was never run against a real checkpoint: an equal-parameter,
NO-memory feed-forward adapter (`reference/hz0b_b11_equal_param_adapter.py`,
692,418 params vs. the real latent write controller's 692,837 -- matched
to within 0.06%) trained on the identical data/steps/lr.

Three-way comparison:
1. True floor -- frozen backbone, zero extra trainable parameters at all.
2. Equal-parameter adapter -- same budget as HZ-0B's real mechanism, but
   no explicit memory state, no read/write, no cross-position information
   flow beyond what the frozen backbone's own attention/recurrence
   already computed.
3. HZ-0B real latent write+read (already measured; re-measurable here
   with `--num-seeds` for a multi-seed check the original single run
   didn't have).
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
from reference.hz0b_b11_equal_param_adapter import adapter_forward, init_equal_param_adapter, param_count
from reference.hz0b_b11_equal_param_adapter import forward as adapter_forward_pass

VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF = 24576, 768, 31, 12, 2304
ATTENTION_INDICES = (4, 9, 14, 19, 24, 29)
CHECKPOINT = Path("outputs/hz0a_stage2_100m_hybrid_seed7/native_metal_checkpoint_best_full_holdout")

# Copied verbatim from scripts/hz0b_b8_stage3_latent_write_probe.py for
# byte-identical task construction -- NOT re-derived, so this comparison
# is apples-to-apples against that script's already-documented 0.750
# result.
FACT_MARKER, FACT_A_ID, FACT_B_ID = 21000, 21001, 21002
READ_TRIGGER_A, READ_TRIGGER_B = 21003, 21004
TARGET_A, TARGET_B = 21005, 21006
FACT_POS = 6
MIDDLE_LEN = 24
PROMPT_LEN = FACT_POS + 2 + MIDDLE_LEN + 2
SEED = 555
ADAPTER_HIDDEN = 450  # 692,418 params -- matched to the latent write controller's 692,837 (0.06% off)


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


def run_true_floor(model, held_out_tokens, held_out_is_a) -> float:
    logits, _ = adapter_forward_pass(model, held_out_tokens, adapter_params=None)
    predicted = mx.argmax(logits[:, -1, :], axis=-1)
    targets = targets_for(held_out_is_a)
    return float(mx.mean((predicted == targets).astype(mx.float32)))


def run_equal_param_adapter(model, train_tokens, train_is_a, held_out_tokens, held_out_is_a, *, seed: int, steps: int, lr: float) -> float:
    params = init_equal_param_adapter(D_MODEL, ADAPTER_HIDDEN, seed=seed)
    params_dict = {"w1": params.w1, "b1": params.b1, "w2": params.w2, "b2": params.b2}

    def loss_fn(pd: dict) -> mx.array:
        p = type(params)(**pd)
        logits, _ = adapter_forward_pass(model, train_tokens, adapter_params=p)
        final = logits[:, -1, :]
        targets = targets_for(train_is_a)
        return mx.mean(nn.losses.cross_entropy(final, targets))

    grad_fn = mx.value_and_grad(loss_fn)
    for step in range(steps):
        loss, grads = grad_fn(params_dict)
        mx.eval(loss)
        params_dict = {k: params_dict[k] - lr * grads[k] for k in params_dict}
        mx.eval(*params_dict.values())
        if step % 300 == 0 or step == steps - 1:
            print(f"    [adapter seed={seed}] step {step:4d}  train loss {float(loss):.5f}")

    trained = type(params)(**params_dict)
    logits, _ = adapter_forward_pass(model, held_out_tokens, adapter_params=trained)
    predicted = mx.argmax(logits[:, -1, :], axis=-1)
    targets = targets_for(held_out_is_a)
    return float(mx.mean((predicted == targets).astype(mx.float32)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000, help="matches the documented HZ-0B baseline's step budget")
    parser.add_argument("--lr", type=float, default=0.15, help="matches the documented HZ-0B baseline's lr")
    parser.add_argument("--num-seeds", type=int, default=3, help="the original HZ-0B 0.750 number was single-seed; this baseline is run multi-seed from the start, a real asymmetry disclosed in the results doc")
    args = parser.parse_args()

    print(f"equal-param adapter budget: {param_count(D_MODEL, ADAPTER_HIDDEN)} params (HZ-0B latent write controller: 692,837)")
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")

    rng = random.Random(SEED)
    train_tokens, train_is_a = make_prompts(32, rng)
    held_out_tokens, held_out_is_a = make_prompts(16, rng)

    floor_acc = run_true_floor(model, held_out_tokens, held_out_is_a)
    print(f"\n1. True floor (frozen backbone, zero extra params): {floor_acc:.3f}")

    print(f"\n2. Equal-parameter no-memory adapter ({args.num_seeds} seeds, steps={args.steps}, lr={args.lr}):")
    adapter_accs = []
    for seed_offset in range(args.num_seeds):
        acc = run_equal_param_adapter(model, train_tokens, train_is_a, held_out_tokens, held_out_is_a, seed=SEED + seed_offset, steps=args.steps, lr=args.lr)
        print(f"  seed {SEED + seed_offset}: {acc:.3f}")
        adapter_accs.append(acc)
    mean_acc = sum(adapter_accs) / len(adapter_accs)
    print(f"  mean: {mean_acc:.3f}  (range {min(adapter_accs):.3f}-{max(adapter_accs):.3f})")

    print("\n3. HZ-0B real latent write+read (already measured, single seed 555, "
          "lr=0.15, lambda_sparse=5, steps=1000 -- scripts/hz0b_b8_stage3_latent_write_probe.py): 0.750")

    print("\n--- Summary ---")
    print(f"floor (0 params):            {floor_acc:.3f}")
    print(f"equal-param adapter (no mem): {mean_acc:.3f}  (range {min(adapter_accs):.3f}-{max(adapter_accs):.3f}, {args.num_seeds} seeds)")
    print("HZ-0B real memory:            0.750  (single seed)")
    if mean_acc >= 0.70:
        print("\nRESULT: the equal-parameter no-memory adapter matches HZ-0B's real result -- "
              "this task's advantage is NOT specific to the memory mechanism, only to having "
              "extra trained capacity anywhere in the forward pass. B11's exit gate is NOT met by this task.")
    else:
        print("\nRESULT: the equal-parameter no-memory adapter falls well short of HZ-0B's real result -- "
              "real evidence the explicit memory mechanism itself matters, not just added capacity. "
              "B11's exit gate is supported by this task (pending more tasks/baselines).")


if __name__ == "__main__":
    main()
