"""HZ-0B B11: root-cause diagnosis for code-symbol tracking's negative
result (docs/restart/hz0b_b11_code_symbol_tracking_results.md). That
doc named an unverified hypothesis: "the controller may be writing/
blending all 3 reassignments rather than cleanly overwriting." This
directly tests it by training ONE real instance (same config as
scripts/hz0b_b11_code_symbol_tracking.py) and then manually stepping
through a held-out example position-by-position, calling
`_choose_write_slot` at each of the 3 ASSIGN_MARKER value positions to
see which slot each reassignment's write actually targets: the SAME
slot each time (a real in-place overwrite -- the mechanism would then
be a clean overwrite, and the failure must be a read-time confusion
issue instead) or 3 DIFFERENT slots (confirming the named hypothesis:
each reassignment is stored as an independent, competing entry, not an
overwrite of the previous one).
"""
from __future__ import annotations

import random

import mlx.core as mx
import mlx.nn as nn

from reference.hz0b_b6_hz0a_integration import frozen_hidden_states
from reference.hz0b_b8_latent_write import latent_write_and_read_step, init_latent_write_controller
from reference.hz0b_memory_simulator import MemoryState, _choose_write_slot, _cosine_similarity
from reference.hz0b_readonly_integration import gated_memory_read
from scripts.hz0b_b11_code_symbol_tracking import (
    ASSIGN_MARKER, LAMBDA_SPARSE, NUM_SLOTS, PROMPT_LEN, READ_TRIGGER, SEED, TARGET_WRITE_RATE,
    dict_to_latent_params, latent_params_to_dict, load_frozen_model, make_prompts, targets_for,
)

STEPS = 1000
LR = 0.15
TRAIN_COUNT = 80


def main():
    print("Training one real instance (seed 555) with the exact code-symbol-tracking config...")
    model, payload = load_frozen_model()
    rng = random.Random(SEED)
    train_tokens, train_idx = make_prompts(TRAIN_COUNT, rng)
    held_out_tokens, held_out_idx = make_prompts(8, rng)
    train_hidden, _ = frozen_hidden_states(model, train_tokens)
    held_out_hidden, _ = frozen_hidden_states(model, held_out_tokens)
    mx.eval(train_hidden, held_out_hidden)

    from reference.hz0b_b8_latent_write import forward as latent_forward_pass

    init_params = init_latent_write_controller(768, 32, 32, seed=SEED)
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
    for step in range(STEPS):
        loss, grads = grad_fn(params_dict)
        mx.eval(loss)
        params_dict = {k: params_dict[k] - LR * grads[k] for k in params_dict}
        mx.eval(*params_dict.values())
        if step % 300 == 0 or step == STEPS - 1:
            print(f"  step {step:4d}  train loss {float(loss):.5f}")

    trained = dict_to_latent_params(params_dict)

    print(f"\nStepping through {held_out_hidden.shape[0]} held-out examples position-by-position, "
          f"logging slot choice + write_gate at each ASSIGN_MARKER value position...\n")

    # Recover the 3 ASSIGN_MARKER value-token positions from one example's tokens directly.
    assign_positions = [i for i in range(PROMPT_LEN) if int(held_out_tokens[0, i]) == ASSIGN_MARKER]
    value_positions = [p + 1 for p in assign_positions]
    print(f"ASSIGN_MARKER positions: {assign_positions}  (value tokens at {value_positions})\n")

    same_slot_count = 0
    all_gates = []
    read_focus_flags = []
    read_weights_on_target = []
    for ex in range(held_out_hidden.shape[0]):
        memory_state = MemoryState(
            keys=mx.zeros((1, NUM_SLOTS, 32)), values=mx.zeros((1, NUM_SLOTS, 32)),
            confidence=mx.zeros((1, NUM_SLOTS)), age=mx.zeros((1, NUM_SLOTS), dtype=mx.int32),
            protection=mx.zeros((1, NUM_SLOTS)), write_count=mx.zeros((1, NUM_SLOTS), dtype=mx.int32),
            last_write_step=mx.zeros((1, NUM_SLOTS), dtype=mx.int32), write_source=mx.zeros((1, NUM_SLOTS), dtype=mx.int32),
        )
        chosen_slots, gates_at_value_pos, keys_at_value_pos = [], [], []
        example_hidden = held_out_hidden[ex:ex + 1]
        read_trigger_pos = PROMPT_LEN - 1
        assert int(held_out_tokens[ex, read_trigger_pos]) == READ_TRIGGER
        pre_read_trigger_state = None
        for t in range(PROMPT_LEN):
            hidden_t = example_hidden[:, t, :]
            key = hidden_t @ trained.key_proj_w + trained.key_proj_b
            if t in value_positions:
                slot_idx, rejected = _choose_write_slot(memory_state, key)
                write_logit = (hidden_t @ trained.write_controller.write_gate_w + trained.write_controller.write_gate_b)[:, 0] \
                    + mx.max(memory_state.confidence, axis=-1) * trained.occupancy_gate_w[0]
                gate = float(mx.sigmoid(write_logit)[0])
                chosen_slots.append(int(slot_idx[0]))
                gates_at_value_pos.append(gate)
                keys_at_value_pos.append(key)
            if t == read_trigger_pos:
                pre_read_trigger_state = memory_state  # read uses the PRE-write state at this position
            _, memory_state, _ = latent_write_and_read_step(trained, hidden_t, memory_state, step=t)

        _, read_weights = gated_memory_read(trained.write_controller.read_params, example_hidden[:, read_trigger_pos, :], pre_read_trigger_state)
        target_slot = chosen_slots[-1] if chosen_slots else None
        read_weight_on_target_slot = float(read_weights[0, target_slot]) if target_slot is not None else float("nan")
        read_weight_argmax = int(mx.argmax(read_weights[0]))
        print(f"  [read diagnosis] final memory slot holding the LAST reassignment's value: slot {target_slot}")
        print(f"  [read diagnosis] read_weights argmax slot: {read_weight_argmax}  weight on that slot: {float(mx.max(read_weights[0])):.4f}  weight on target slot {target_slot}: {read_weight_on_target_slot:.4f}")

        pairwise_key_sims = []
        for i in range(len(keys_at_value_pos)):
            for j in range(i + 1, len(keys_at_value_pos)):
                sim = float(_cosine_similarity(keys_at_value_pos[i], keys_at_value_pos[j])[0])
                pairwise_key_sims.append(((i, j), sim))

        all_same_slot = len(set(chosen_slots)) == 1
        same_slot_count += int(all_same_slot)
        all_gates.extend(gates_at_value_pos)
        read_correctly_focused = (read_weight_argmax == target_slot)
        read_focus_flags.append(read_correctly_focused)
        read_weights_on_target.append(read_weight_on_target_slot)
        print(f"example {ex}: chosen slots at 3 reassignments = {chosen_slots}  "
              f"({'SAME slot every time (real overwrite)' if all_same_slot else 'DIFFERENT slots (separate entries, not an overwrite)'})")
        print(f"  write_gate at each reassignment: {[f'{g:.3f}' for g in gates_at_value_pos]}")
        print(f"  pairwise key cosine similarity between reassignments: {[(pair, f'{s:.4f}') for pair, s in pairwise_key_sims]}")

    n = held_out_hidden.shape[0]
    print(f"\n--- Summary ---")
    print(f"examples where all 3 reassignments hit the SAME slot (real overwrite): {same_slot_count}/{n}")
    print(f"examples where reassignments were split across DIFFERENT slots: {n - same_slot_count}/{n}")
    print(f"mean write_gate at reassignment positions: {sum(all_gates)/len(all_gates):.3f}  (range {min(all_gates):.3f}-{max(all_gates):.3f})")
    print(f"examples where the READ at READ_TRIGGER correctly focused (argmax) on the slot holding the final value: {sum(read_focus_flags)}/{n}")
    print(f"mean read_weight placed on the correct (final-value) slot: {sum(read_weights_on_target)/len(read_weights_on_target):.4f}  (range {min(read_weights_on_target):.4f}-{max(read_weights_on_target):.4f})")


if __name__ == "__main__":
    main()
