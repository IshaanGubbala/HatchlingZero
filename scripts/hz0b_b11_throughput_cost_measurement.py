"""HZ-0B B11: real throughput-under-load + end-to-end cost measurement.

Two of B11's 16 named eval tasks were still unstarted: "throughput
under load" and "end-to-end cost measurements". Both are infra/timing
questions, not accuracy questions -- measured directly here against
the real frozen HZ-0A checkpoint, real memory controller, real equal-
param adapter. No synthetic task-design risk (the failure mode that
bit earlier accuracy-task scripts) since nothing here depends on the
task's semantic content, only its shape (sequence length, batch size).

Reports real wall-clock (ms/step, tokens/sec) and real peak memory
(mx.metal.get_peak_memory) for:
  1. one-time backbone forward cost (the thing the 2026-08-01 caching
     optimization amortizes across all training steps)
  2. per-step train cost (forward + backward + update) for the memory
     controller vs the equal-param adapter, at num_slots in {4, 8, 16}
     ("load" = more slots = more per-position write/read work)
  3. projected full-run wall-clock (steps x per-step cost) so the
     caching win has a real number attached instead of "should help"
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from reference.hz0a_mlx_model import HZ0AMlxModel
from reference.hz0b_b6_hz0a_integration import frozen_hidden_states
from reference.hz0b_b8_latent_write import forward as latent_forward_pass, init_latent_write_controller
from reference.hz0b_b11_equal_param_adapter import forward as adapter_forward_pass, init_equal_param_adapter, param_count
from scripts.hz0b_b11_passkey_task import latent_params_to_dict, dict_to_latent_params

VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF = 24576, 768, 31, 12, 2304
ATTENTION_INDICES = (4, 9, 14, 19, 24, 29)
CHECKPOINT = Path("outputs/hz0a_stage2_100m_hybrid_seed7/native_metal_checkpoint_best_full_holdout")
FACT_MARKER, FACT_ID, READ_TRIGGER, TARGET = 21000, 21001, 21002, 21003
FACT_POS, MIDDLE_LEN = 6, 24
PROMPT_LEN = FACT_POS + 2 + MIDDLE_LEN + 1
ADAPTER_HIDDEN = 450
KEY_DIM = VALUE_DIM = 32
LAMBDA_SPARSE = 0.1
SEED = 555


def load_frozen_model():
    payload = json.loads((CHECKPOINT / "state.json").read_text())
    model = HZ0AMlxModel(VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF, ATTENTION_INDICES, native_metal=True)
    model_arrays = [(item["key"], mx.load(str(CHECKPOINT / item["file"]))) for item in payload["arrays"] if item["group"] == "model"]
    model.update(tree_unflatten(model_arrays))
    mx.eval(model.parameters())
    return model


def make_prompts(count: int, rng: random.Random) -> mx.array:
    rows = []
    for _ in range(count):
        prefix = [rng.randint(100, VOCAB_SIZE - 100) for _ in range(FACT_POS)]
        middle = [rng.randint(100, VOCAB_SIZE - 100) for _ in range(MIDDLE_LEN)]
        rows.append(prefix + [FACT_MARKER, FACT_ID] + middle + [READ_TRIGGER])
    return mx.array(rows, dtype=mx.int32)


def time_backbone_forward(model, tokens: mx.array, reps: int) -> float:
    for _ in range(2):
        h, _ = frozen_hidden_states(model, tokens)
        mx.eval(h)
    t0 = time.perf_counter()
    for _ in range(reps):
        h, _ = frozen_hidden_states(model, tokens)
        mx.eval(h)
    return (time.perf_counter() - t0) / reps


def time_adapter_step(model, hidden: mx.array, targets: mx.array, *, steps: int, warmup: int, lr: float = 0.15) -> tuple[float, int]:
    params = init_equal_param_adapter(D_MODEL, ADAPTER_HIDDEN, seed=SEED)
    pd = {"w1": params.w1, "b1": params.b1, "w2": params.w2, "b2": params.b2}

    def loss_fn(pd):
        p = type(params)(**pd)
        logits, _ = adapter_forward_pass(model, precomputed_hidden=hidden, adapter_params=p)
        return mx.mean(nn.losses.cross_entropy(logits[:, -1, :], targets))

    grad_fn = mx.value_and_grad(loss_fn)
    for _ in range(warmup):
        loss, grads = grad_fn(pd)
        mx.eval(loss)
        pd = {k: pd[k] - lr * grads[k] for k in pd}
        mx.eval(*pd.values())

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for _ in range(steps):
        loss, grads = grad_fn(pd)
        mx.eval(loss)
        pd = {k: pd[k] - lr * grads[k] for k in pd}
        mx.eval(*pd.values())
    elapsed = time.perf_counter() - t0
    peak = mx.get_peak_memory()
    return elapsed / steps, peak


def time_memory_step(model, hidden: mx.array, targets: mx.array, *, num_slots: int, steps: int, warmup: int, lr: float = 0.15) -> tuple[float, int]:
    init_params = init_latent_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=SEED)
    pd = latent_params_to_dict(init_params)

    def loss_fn(pd):
        p = dict_to_latent_params(pd)
        logits, _, gates = latent_forward_pass(model, precomputed_hidden=hidden, latent_params=p, num_slots=num_slots)
        task_loss = mx.mean(nn.losses.cross_entropy(logits[:, -1, :], targets))
        return task_loss + LAMBDA_SPARSE * mx.mean(gates)

    grad_fn = mx.value_and_grad(loss_fn)
    for _ in range(warmup):
        loss, grads = grad_fn(pd)
        mx.eval(loss)
        pd = {k: pd[k] - lr * grads[k] for k in pd}
        mx.eval(*pd.values())

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for _ in range(steps):
        loss, grads = grad_fn(pd)
        mx.eval(loss)
        pd = {k: pd[k] - lr * grads[k] for k in pd}
        mx.eval(*pd.values())
    elapsed = time.perf_counter() - t0
    peak = mx.get_peak_memory()
    return elapsed / steps, peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--backbone-reps", type=int, default=5)
    parser.add_argument("--full-run-steps", type=int, default=1000, help="for the projected-cost table")
    args = parser.parse_args()

    print("loading frozen HZ-0A checkpoint...")
    model = load_frozen_model()
    rng = random.Random(SEED)
    tokens = make_prompts(args.batch, rng)
    targets = mx.full((args.batch,), TARGET, dtype=mx.int32)
    print(f"batch={args.batch} seq_len={PROMPT_LEN} steps/config={args.steps} warmup={args.warmup}\n")

    backbone_time = time_backbone_forward(model, tokens, args.backbone_reps)
    backbone_tok_s = (args.batch * PROMPT_LEN) / backbone_time
    print(f"1. Backbone forward (301M frozen, one-time per dataset): {backbone_time*1000:.1f} ms/call  ({backbone_tok_s:,.0f} tok/s)")
    print(f"   Amortized cost if recomputed every step for {args.full_run_steps} steps (the pre-caching behavior): {backbone_time*args.full_run_steps:.1f}s")
    print(f"   Actual cost with caching (computed once): {backbone_time:.3f}s -- saves {backbone_time*(args.full_run_steps-1):.1f}s per {args.full_run_steps}-step run\n")

    hidden, _ = frozen_hidden_states(model, tokens)
    mx.eval(hidden)

    print("2. Per-step train cost (forward+backward+update), precomputed hidden (post-caching):")
    adapter_ms, adapter_peak = time_adapter_step(model, hidden, targets, steps=args.steps, warmup=args.warmup)
    print(f"   equal-param adapter:            {adapter_ms*1000:.2f} ms/step  peak mem {adapter_peak/1e6:.1f} MB")

    results = []
    for num_slots in (4, 8, 16):
        ms, peak = time_memory_step(model, hidden, targets, num_slots=num_slots, steps=args.steps, warmup=args.warmup)
        results.append((num_slots, ms, peak))
        print(f"   memory controller (num_slots={num_slots:2d}): {ms*1000:.2f} ms/step  peak mem {peak/1e6:.1f} MB  ({ms/adapter_ms:.1f}x adapter)")

    print(f"\n3. Projected full {args.full_run_steps}-step training run wall-clock (memory-step cost only, backbone already cached):")
    print(f"   equal-param adapter: {adapter_ms*args.full_run_steps:.1f}s")
    for num_slots, ms, _ in results:
        print(f"   memory (num_slots={num_slots:2d}): {ms*args.full_run_steps:.1f}s")


if __name__ == "__main__":
    main()
