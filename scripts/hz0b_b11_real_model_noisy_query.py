"""HZ-0B B11: noisy associative recall -- one of the plan's 16 named
eval tasks, not yet covered by a real-model test. B8 Stage 5's own
"noisy query" scenario (`scripts/hz0b_b11_stage5_baseline_comparison.py`)
perturbs a raw key vector directly (`noisy_query = key + gaussian noise,
renormalized`) since the pure simulator has no language model to route
noise through. This is the real-model analog: instead of perturbing an
abstract key vector, Gaussian noise is injected directly into the
REAL frozen backbone's hidden state at the read-trigger position
before it reaches the memory read query -- the natural real-model
counterpart of "the query is an imprecise/noisy version of the
original cue", since in the real pipeline the query is `hidden_state
@ query_w + query_b`, not a hand-picked vector.

Reuses `scripts/hz0b_b11_baseline_comparison.py`'s exact task and the
validated `lambda_sparse=0.1` + `target_write_rate=0.1` config.
Trains ONCE per seed on CLEAN data (matching how a real deployed
memory would be trained), then evaluates held-out accuracy at several
noise levels injected only at eval time, only at the read-trigger
position -- measuring degradation as a function of noise, not just a
single pass/fail point.
"""
from __future__ import annotations

import argparse
import random

import mlx.core as mx
import mlx.nn as nn

from reference.hz0b_b6_hz0a_integration import frozen_hidden_states
from reference.hz0b_b8_latent_write import forward as latent_forward_pass, init_latent_write_controller
from reference.hz0b_b11_equal_param_adapter import forward as adapter_forward_pass, init_equal_param_adapter
from scripts.hz0b_b11_baseline_comparison import (
    ADAPTER_HIDDEN, D_MODEL, KEY_DIM, NUM_SLOTS, SEED, VALUE_DIM,
    dict_to_latent_params, latent_params_to_dict, load_frozen_model, make_prompts, targets_for,
)

LAMBDA_SPARSE = 0.1  # the validated fix, NOT hz0b_b11_baseline_comparison's stale 5.0 default -- see docs/restart/hz0b_b11_evaluation_results.md
TARGET_WRITE_RATE = 0.1
NOISE_LEVELS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]


def inject_noise_at_last_position(hidden: mx.array, scale: float, seed: int) -> mx.array:
    """hidden: [batch, seq, d_model]. Adds Gaussian noise scaled by
    `scale * per-example hidden-state std` ONLY at the final (read-
    trigger) position -- the position whose hidden state becomes the
    memory read query. Every other position (including the write
    positions) is untouched, matching the simulator scenario's own
    scope (only the QUERY is noisy, not the stored key/value)."""
    if scale == 0.0:
        return hidden
    mx.random.seed(seed)
    last = hidden[:, -1, :]
    std = mx.std(last, axis=-1, keepdims=True)
    noise = mx.random.normal(last.shape) * std * scale
    noisy_last = last + noise
    return mx.concatenate([hidden[:, :-1, :], noisy_last[:, None, :]], axis=1)


def eval_at_noise(forward_fn, hidden, idx, *, noise_scale: float, seed_for_noise: int) -> float:
    noisy_hidden = inject_noise_at_last_position(hidden, noise_scale, seed_for_noise)
    logits = forward_fn(noisy_hidden)
    predicted = mx.argmax(logits[:, -1, :], axis=-1)
    return float(mx.mean((predicted == targets_for(idx)).astype(mx.float32)))


def train_adapter(model, train_hidden, train_is_a, *, seed: int, steps: int, lr: float):
    params = init_equal_param_adapter(D_MODEL, ADAPTER_HIDDEN, seed=seed)
    params_dict = {"w1": params.w1, "b1": params.b1, "w2": params.w2, "b2": params.b2}
    targets = targets_for(train_is_a)

    def loss_fn(pd):
        p = type(params)(**pd)
        logits, _ = adapter_forward_pass(model, precomputed_hidden=train_hidden, adapter_params=p)
        return mx.mean(nn.losses.cross_entropy(logits[:, -1, :], targets))

    grad_fn = mx.value_and_grad(loss_fn)
    for step in range(steps):
        loss, grads = grad_fn(params_dict)
        mx.eval(loss)
        params_dict = {k: params_dict[k] - lr * grads[k] for k in params_dict}
        mx.eval(*params_dict.values())

    trained = type(params)(**params_dict)
    return lambda h: adapter_forward_pass(model, precomputed_hidden=h, adapter_params=trained)[0]


def train_memory(model, train_hidden, train_is_a, *, seed: int, steps: int, lr: float):
    init_params = init_latent_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=seed)
    params_dict = latent_params_to_dict(init_params)
    targets = targets_for(train_is_a)

    def loss_fn(pd):
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

    trained = dict_to_latent_params(params_dict)
    return lambda h: latent_forward_pass(model, precomputed_hidden=h, latent_params=trained, num_slots=NUM_SLOTS)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--held-out-count", type=int, default=64)
    args = parser.parse_args()

    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")

    rng = random.Random(SEED)
    train_tokens, train_is_a = make_prompts(args.train_count, rng)
    held_out_tokens, held_out_is_a = make_prompts(args.held_out_count, rng)
    print(f"train_count={args.train_count} held_out_count={args.held_out_count} noise_levels={NOISE_LEVELS}")

    train_hidden, _ = frozen_hidden_states(model, train_tokens)
    held_out_hidden, _ = frozen_hidden_states(model, held_out_tokens)
    mx.eval(train_hidden, held_out_hidden)

    print(f"\nAdapter ({args.num_seeds} seeds) x noise level:")
    adapter_by_noise = {n: [] for n in NOISE_LEVELS}
    for i in range(args.num_seeds):
        fwd = train_adapter(model, train_hidden, train_is_a, seed=SEED + i, steps=args.steps, lr=args.lr)
        for n in NOISE_LEVELS:
            acc = eval_at_noise(fwd, held_out_hidden, held_out_is_a, noise_scale=n, seed_for_noise=1000 + i)
            adapter_by_noise[n].append(acc)
        print(f"  seed {SEED + i}: " + "  ".join(f"noise={n}: {adapter_by_noise[n][-1]:.3f}" for n in NOISE_LEVELS))

    print(f"\nMemory ({args.num_seeds} seeds) x noise level:")
    memory_by_noise = {n: [] for n in NOISE_LEVELS}
    for i in range(args.num_seeds):
        fwd = train_memory(model, train_hidden, train_is_a, seed=SEED + i, steps=args.steps, lr=args.lr)
        for n in NOISE_LEVELS:
            acc = eval_at_noise(fwd, held_out_hidden, held_out_is_a, noise_scale=n, seed_for_noise=1000 + i)
            memory_by_noise[n].append(acc)
        print(f"  seed {SEED + i}: " + "  ".join(f"noise={n}: {memory_by_noise[n][-1]:.3f}" for n in NOISE_LEVELS))

    print("\n--- Summary (mean accuracy vs. noise level) ---")
    print(f"{'noise':>8}  {'adapter_mean':>12}  {'memory_mean':>12}")
    for n in NOISE_LEVELS:
        a_mean = sum(adapter_by_noise[n]) / len(adapter_by_noise[n])
        m_mean = sum(memory_by_noise[n]) / len(memory_by_noise[n])
        print(f"{n:>8.2f}  {a_mean:>12.3f}  {m_mean:>12.3f}")


if __name__ == "__main__":
    main()
