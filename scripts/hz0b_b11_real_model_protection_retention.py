"""HZ-0B B11: protection retention -- one of the plan's 16 named eval
tasks, not yet covered by a real-model test. B8 Stage 5's own
"protected-memory overwrite rejection" scenario is pure-simulator
(direct `write()`/`protect()` calls, oracle slot targeting, no LM, no
learned write timing/slot choice). This tests whether a PROTECTED
slot survives real, LEARNED write pressure: slot 0 is pre-populated
with a synthetic "anchor" fact and `protect()`-ed to full strength
BEFORE training starts; the real latent write controller is then
trained on an ordinary single-fact recall task using the REMAINING 7
slots. Checks, after training, on real held-out examples: (1) slot
0's key/value/confidence/protection are bit-identical to their
pre-training values (never touched by any learned write), and (2) the
controller still solves its own task using the remaining capacity.

Manually replicates the sequential write/read loop (like
scripts/hz0b_b11_write_slot_diagnosis_code_symbol.py) instead of
`sequential_latent_write_and_read`, since that helper always starts
from an all-zero memory state -- there is no opt-in "initial state"
param, and adding one to the shared reference module for a single
diagnostic script would be more invasive than replicating the loop
locally.
"""
from __future__ import annotations

import random

import mlx.core as mx
import mlx.nn as nn

from reference.hz0b_b6_hz0a_integration import frozen_hidden_states
from reference.hz0b_b8_latent_write import latent_write_and_read_step, init_latent_write_controller
from reference.hz0b_memory_simulator import MemoryState, protect as memory_protect, write as memory_write
from scripts.hz0b_b11_baseline_comparison import (
    D_MODEL, KEY_DIM, NUM_SLOTS, SEED, VALUE_DIM,
    dict_to_latent_params, latent_params_to_dict, load_frozen_model, make_prompts, targets_for,
)

LAMBDA_SPARSE = 0.1
TARGET_WRITE_RATE = 0.1
STEPS = 1000
LR = 0.15
NUM_SEEDS = 5
TRAIN_COUNT = 64
HELD_OUT_COUNT = 64
PROTECTED_SLOT = 0


def make_protected_initial_state(batch: int) -> MemoryState:
    """A synthetic anchor fact in slot 0, protected to full strength,
    unrelated to the real task's own facts (fixed one-hot-ish key/value
    so it is trivially distinguishable from anything the real task
    could plausibly produce)."""
    state = MemoryState(
        keys=mx.zeros((batch, NUM_SLOTS, KEY_DIM)), values=mx.zeros((batch, NUM_SLOTS, VALUE_DIM)),
        confidence=mx.zeros((batch, NUM_SLOTS)), age=mx.zeros((batch, NUM_SLOTS), dtype=mx.int32),
        protection=mx.zeros((batch, NUM_SLOTS)), write_count=mx.zeros((batch, NUM_SLOTS), dtype=mx.int32),
        last_write_step=mx.zeros((batch, NUM_SLOTS), dtype=mx.int32), write_source=mx.zeros((batch, NUM_SLOTS), dtype=mx.int32),
    )
    anchor_key = mx.ones((batch, KEY_DIM)) * 0.7071  # fixed, arbitrary unit-ish vector
    anchor_value = mx.ones((batch, VALUE_DIM)) * -0.7071
    state, _, rejected = memory_write(state, anchor_key, anchor_value, mx.array([1.0] * batch), step=-1, slot_idx=mx.array([PROTECTED_SLOT] * batch))
    assert not bool(mx.any(rejected))
    state = memory_protect(state, mx.array([PROTECTED_SLOT] * batch), mx.array([1.0] * batch))
    return state


def sequential_forward_from_state(params, hidden: mx.array, initial_state: MemoryState):
    batch, seq, _ = hidden.shape
    memory_state = initial_state
    outputs, gates = [], []
    for t in range(seq):
        output, memory_state, write_gate = latent_write_and_read_step(params, hidden[:, t, :], memory_state, step=t)
        outputs.append(output)
        gates.append(write_gate)
    return mx.stack(outputs, axis=1), memory_state, mx.stack(gates, axis=1)


def run_seed(model, train_hidden, train_is_a, held_out_hidden, held_out_is_a, *, seed: int):
    init_params = init_latent_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=seed)
    params_dict = latent_params_to_dict(init_params)
    targets = targets_for(train_is_a)
    train_batch = train_hidden.shape[0]

    def loss_fn(pd):
        p = dict_to_latent_params(pd)
        protected_state = make_protected_initial_state(train_batch)
        out_hidden, _, gates = sequential_forward_from_state(p, train_hidden, protected_state)
        from reference.hz0b_b6_hz0a_integration import logits_from_hidden
        logits = logits_from_hidden(model, out_hidden)
        task_loss = mx.mean(nn.losses.cross_entropy(logits[:, -1, :], targets))
        write_rate = mx.mean(gates)
        sparsity_loss = (write_rate - TARGET_WRITE_RATE) ** 2
        return task_loss + LAMBDA_SPARSE * sparsity_loss

    grad_fn = mx.value_and_grad(loss_fn)
    for step in range(STEPS):
        loss, grads = grad_fn(params_dict)
        mx.eval(loss)
        params_dict = {k: params_dict[k] - LR * grads[k] for k in params_dict}
        mx.eval(*params_dict.values())
        if step % 300 == 0 or step == STEPS - 1:
            print(f"    [seed={seed}] step {step:4d}  train loss {float(loss):.5f}")

    trained = dict_to_latent_params(params_dict)
    held_out_batch = held_out_hidden.shape[0]
    protected_state = make_protected_initial_state(held_out_batch)
    out_hidden, final_state, _ = sequential_forward_from_state(trained, held_out_hidden, protected_state)
    from reference.hz0b_b6_hz0a_integration import logits_from_hidden
    logits = logits_from_hidden(model, out_hidden)
    predicted = mx.argmax(logits[:, -1, :], axis=-1)
    acc = float(mx.mean((predicted == targets_for(held_out_is_a)).astype(mx.float32)))

    fresh_protected = make_protected_initial_state(held_out_batch)
    # Per-EXAMPLE intactness (not a single whole-batch mx.array_equal,
    # which reports "changed" for every example the instant even ONE
    # example out of the batch differs -- a real, misleading bug in an
    # earlier version of this script, caught after the whole-batch
    # check reported "not intact" for 4/5 seeds and a direct per-
    # example audit found only 1/64 examples actually affected).
    key_changed_per_example = mx.any(final_state.keys[:, PROTECTED_SLOT, :] != fresh_protected.keys[:, PROTECTED_SLOT, :], axis=-1)
    value_changed_per_example = mx.any(final_state.values[:, PROTECTED_SLOT, :] != fresh_protected.values[:, PROTECTED_SLOT, :], axis=-1)
    any_changed_per_example = key_changed_per_example | value_changed_per_example
    n_changed = int(mx.sum(any_changed_per_example.astype(mx.float32)))
    confidence_intact = bool(mx.array_equal(final_state.confidence[:, PROTECTED_SLOT], fresh_protected.confidence[:, PROTECTED_SLOT]))
    protection_field_intact = bool(mx.array_equal(final_state.protection[:, PROTECTED_SLOT], fresh_protected.protection[:, PROTECTED_SLOT]))
    return acc, n_changed, held_out_batch, confidence_intact, protection_field_intact


def main():
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")

    rng = random.Random(SEED)
    train_tokens, train_is_a = make_prompts(TRAIN_COUNT, rng)
    held_out_tokens, held_out_is_a = make_prompts(HELD_OUT_COUNT, rng)
    print(f"train_count={TRAIN_COUNT} held_out_count={HELD_OUT_COUNT} num_slots={NUM_SLOTS} protected_slot={PROTECTED_SLOT} (7 slots usable by the learned controller)")

    train_hidden, _ = frozen_hidden_states(model, train_tokens)
    held_out_hidden, _ = frozen_hidden_states(model, held_out_tokens)
    mx.eval(train_hidden, held_out_hidden)

    accs, total_changed, total_examples = [], 0, 0
    for i in range(NUM_SEEDS):
        acc, n_changed, n_examples, conf_ok, prot_ok = run_seed(model, train_hidden, train_is_a, held_out_hidden, held_out_is_a, seed=SEED + i)
        accs.append(acc)
        total_changed += n_changed
        total_examples += n_examples
        print(f"  seed {SEED + i}: task accuracy {acc:.3f}  protected slot leaked on {n_changed}/{n_examples} held-out examples ({100*n_changed/n_examples:.1f}%)  confidence_field_intact={conf_ok} protection_field_intact={prot_ok}")

    mean_acc = sum(accs) / len(accs)
    print(f"\n--- Summary ---")
    print(f"mean task accuracy (using 7 unprotected slots): {mean_acc:.3f}  range {min(accs):.3f}-{max(accs):.3f}")
    print(f"protected slot leaked on {total_changed}/{total_examples} held-out examples total ({100*total_changed/total_examples:.1f}%) across all {NUM_SEEDS} seeds")


if __name__ == "__main__":
    main()
