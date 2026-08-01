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
import dataclasses
import json
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from reference.hz0a_mlx_model import HZ0AMlxModel
from reference.hz0b_b8_latent_write import LatentWriteControllerParams, forward as latent_forward_pass, init_latent_write_controller
from reference.hz0b_b11_equal_param_adapter import adapter_forward, init_equal_param_adapter, param_count
from reference.hz0b_b11_equal_param_adapter import forward as adapter_forward_pass
from reference.hz0b_readonly_integration import ReadOnlyIntegrationParams
from reference.hz0b_write_integration import WriteControllerParams

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
KEY_DIM = VALUE_DIM = 32
NUM_SLOTS = 8
LAMBDA_SPARSE = 5.0  # matches the documented baseline (docs/restart/hz0b_b8_stage3_results.md)


def latent_params_to_dict(p: LatentWriteControllerParams) -> dict:
    """Copied from scripts/hz0b_b8_stage3_latent_write_probe.py for
    byte-identical training mechanics (mx.value_and_grad needs a flat
    dict, not the nested dataclass)."""
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


def run_hzb_memory(model, train_tokens, train_is_a, held_out_tokens, held_out_is_a, *, seed: int, steps: int, lr: float, num_slots: int = NUM_SLOTS, ste: bool = False, lambda_sparse: float = LAMBDA_SPARSE) -> float:
    """Real HZ-0B latent write+read, trained fresh at this seed --
    reuses reference/hz0b_b8_latent_write.py unmodified, same mechanics
    as scripts/hz0b_b8_stage3_latent_write_probe.py. `num_slots`
    parameterized (2026-08-01) to test whether the train_count=64
    reversal (docs/restart/hz0b_b11_evaluation_results.md) is a
    slot-capacity/interference artifact of the default 8 slots against
    a doubled training set, not a fundamental mechanism failure (ruled
    out -- see that doc). `ste` (2026-08-01, same day): tests the real
    candidate fix -- hard/discrete write decisions via a straight-through
    estimator instead of the continuous blend traced as the likely root
    cause. `lambda_sparse` (2026-08-01, same day): the factorial write-
    mechanism diagnosis found the sparsity penalty itself starves the
    timing gate's gradient signal when content offers no
    position-differentiating reward -- lowering it rescued cell 3's
    isolated collapse (0.053 -> 0.281); this tests whether it also
    rescues the FULL learned mechanism."""
    init_params = init_latent_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=seed)
    params_dict = latent_params_to_dict(init_params)

    def loss_fn(pd: dict) -> mx.array:
        p = dict_to_latent_params(pd)
        logits, _, gates = latent_forward_pass(model, train_tokens, latent_params=p, num_slots=num_slots, ste=ste)
        final = logits[:, -1, :]
        targets = targets_for(train_is_a)
        task_loss = mx.mean(nn.losses.cross_entropy(final, targets))
        sparsity_loss = mx.mean(gates)
        return task_loss + lambda_sparse * sparsity_loss

    grad_fn = mx.value_and_grad(loss_fn)
    for step in range(steps):
        loss, grads = grad_fn(params_dict)
        mx.eval(loss)
        params_dict = {k: params_dict[k] - lr * grads[k] for k in params_dict}
        mx.eval(*params_dict.values())
        if step % 300 == 0 or step == steps - 1:
            print(f"    [memory seed={seed} num_slots={num_slots} ste={ste}] step {step:4d}  train loss {float(loss):.5f}")

    trained = dict_to_latent_params(params_dict)
    logits, _, _ = latent_forward_pass(model, held_out_tokens, latent_params=trained, num_slots=num_slots, ste=ste)
    predicted = mx.argmax(logits[:, -1, :], axis=-1)
    targets = targets_for(held_out_is_a)
    return float(mx.mean((predicted == targets).astype(mx.float32)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000, help="matches the documented HZ-0B baseline's step budget")
    parser.add_argument("--lr", type=float, default=0.15, help="matches the documented HZ-0B baseline's lr")
    parser.add_argument("--num-seeds", type=int, default=5, help="now applied to BOTH conditions -- fixes the original single-seed-memory-vs-3-seed-adapter asymmetry")
    parser.add_argument("--train-count", type=int, default=64, help="4x the original 32 -- more training signal for the larger held-out check below")
    parser.add_argument("--held-out-count", type=int, default=64, help="4x the original 16 -- 1/64 granularity instead of 1/16, the main fix for the coarse-accuracy caveat")
    parser.add_argument("--num-slots", type=int, default=NUM_SLOTS, help="memory controller's slot count -- tests whether the train_count=64 reversal is slot-capacity interference, not a fundamental mechanism failure")
    parser.add_argument("--skip-adapter", action="store_true", help="skip the adapter condition (no slots, unaffected by --num-slots/--ste) to save time when only re-checking the memory condition")
    parser.add_argument("--ste", action="store_true", help="hard/discrete write decisions via straight-through estimator (reference/hz0b_b8_latent_write.py) instead of the continuous blend -- the real candidate fix for the reversal, per B1 decision 5's deferred hard-routing experiment")
    parser.add_argument("--lambda-sparse", type=float, default=LAMBDA_SPARSE, help="the factorial diagnosis found this penalty starves the timing gate's gradient when content offers no position-differentiating reward -- lowering it rescued the isolated cell-3 collapse (0.053->0.281); tests whether it also helps the full mechanism")
    args = parser.parse_args()

    print(f"equal-param adapter budget: {param_count(D_MODEL, ADAPTER_HIDDEN)} params, "
          f"latent memory controller budget: {692_837} params")
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")

    rng = random.Random(SEED)
    train_tokens, train_is_a = make_prompts(args.train_count, rng)
    held_out_tokens, held_out_is_a = make_prompts(args.held_out_count, rng)
    print(f"train_count={args.train_count} held_out_count={args.held_out_count} (granularity: 1/{args.held_out_count} = {1/args.held_out_count:.4f})")

    floor_acc = run_true_floor(model, held_out_tokens, held_out_is_a)
    print(f"\n1. True floor (frozen backbone, zero extra params): {floor_acc:.3f}")

    adapter_accs = [None]
    if not args.skip_adapter:
        print(f"\n2. Equal-parameter no-memory adapter ({args.num_seeds} seeds, steps={args.steps}, lr={args.lr}):")
        adapter_accs = []
        for seed_offset in range(args.num_seeds):
            acc = run_equal_param_adapter(model, train_tokens, train_is_a, held_out_tokens, held_out_is_a, seed=SEED + seed_offset, steps=args.steps, lr=args.lr)
            print(f"  seed {SEED + seed_offset}: {acc:.3f}")
            adapter_accs.append(acc)
        adapter_mean = sum(adapter_accs) / len(adapter_accs)
        adapter_std = (sum((a - adapter_mean) ** 2 for a in adapter_accs) / len(adapter_accs)) ** 0.5
        print(f"  mean: {adapter_mean:.3f}  std: {adapter_std:.3f}  (range {min(adapter_accs):.3f}-{max(adapter_accs):.3f})")
    else:
        print("\n2. Equal-parameter no-memory adapter: SKIPPED (--skip-adapter)")
        adapter_mean = adapter_std = float("nan")

    print(f"\n3. HZ-0B real latent write+read ({args.num_seeds} seeds, steps={args.steps}, lr={args.lr}, lambda_sparse={args.lambda_sparse}, num_slots={args.num_slots}, ste={args.ste}):")
    memory_accs = []
    for seed_offset in range(args.num_seeds):
        acc = run_hzb_memory(model, train_tokens, train_is_a, held_out_tokens, held_out_is_a, seed=SEED + seed_offset, steps=args.steps, lr=args.lr, num_slots=args.num_slots, ste=args.ste, lambda_sparse=args.lambda_sparse)
        print(f"  seed {SEED + seed_offset}: {acc:.3f}")
        memory_accs.append(acc)
    memory_mean = sum(memory_accs) / len(memory_accs)
    memory_std = (sum((a - memory_mean) ** 2 for a in memory_accs) / len(memory_accs)) ** 0.5
    print(f"  mean: {memory_mean:.3f}  std: {memory_std:.3f}  (range {min(memory_accs):.3f}-{max(memory_accs):.3f})")

    print(f"\n--- Summary (real frozen HZ-0A checkpoint, num_slots={args.num_slots}) ---")
    print(f"floor (0 params):            {floor_acc:.3f}")
    if not args.skip_adapter:
        print(f"equal-param adapter (no mem): mean {adapter_mean:.3f}  std {adapter_std:.3f}  (range {min(adapter_accs):.3f}-{max(adapter_accs):.3f}, {args.num_seeds} seeds)")
    print(f"HZ-0B real memory:            mean {memory_mean:.3f}  std {memory_std:.3f}  (range {min(memory_accs):.3f}-{max(memory_accs):.3f}, {args.num_seeds} seeds)")
    if not args.skip_adapter:
        if memory_mean - adapter_mean < 0.05:
            print("\nRESULT: the equal-parameter no-memory adapter is within 5 points of HZ-0B's real result at this "
                  "larger, multi-seed scale -- this task's advantage may NOT be specific to the memory mechanism.")
        else:
            print("\nRESULT: the equal-parameter no-memory adapter still falls clearly short of HZ-0B's real result at "
                  "this larger, multi-seed scale -- the gap holds up, not an artifact of small-sample noise.")


if __name__ == "__main__":
    main()
