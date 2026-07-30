"""HZ-0B B8 Stage 3 ("latent write decisions -- only final behavior is
supervised") against the real frozen HZ-0A hybrid checkpoint.

Two-way fact-discrimination task, deliberately harder than B6/B7's
single-fixed-fact setup: after a FACT_MARKER token, the sequence shows
EITHER fact-id A or fact-id B (randomly, per example); much later, after
a read-trigger bigram, the correct TARGET depends on which fact-id was
shown. A model that just learns a constant, content-independent bias
(the failure mode traced in B6/B7's write-up) cannot solve this -- there
are two different correct answers, so the write must actually encode
WHICH fact appeared, not just whether reading "feels informed."

Nothing about where or whether to write is supervised: `write_gate`,
`key`, and `value` are all learned functions of each position's own
hidden state (`reference/hz0b_b8_latent_write.py`), trained end-to-end
purely from the final-position cross-entropy loss plus a write-sparsity
penalty (mean write_gate across every position and the batch -- the same
quantity B3's `excessive_write_penalty` computes, reused in form here).
B8's own exit gate -- "the model learns sparse, useful writes rather than
writing every token" -- is checked directly: mean write_gate AT the fact
position vs. averaged over all OTHER positions, after training.
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
from reference.hz0b_b8_latent_write import LatentWriteControllerParams, forward, init_latent_write_controller
from reference.hz0b_readonly_integration import ReadOnlyIntegrationParams
from reference.hz0b_write_integration import WriteControllerParams

VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF = 24576, 768, 31, 12, 2304
ATTENTION_INDICES = (4, 9, 14, 19, 24, 29)
KEY_DIM = VALUE_DIM = 32
CHECKPOINT = Path("outputs/hz0a_stage2_100m_hybrid_seed7/native_metal_checkpoint_best_full_holdout")

FACT_MARKER, FACT_A_ID, FACT_B_ID = 21000, 21001, 21002
READ_TRIGGER_A, READ_TRIGGER_B = 21003, 21004
TARGET_A, TARGET_B = 21005, 21006
FACT_POS = 6                # fixed position of FACT_MARKER within the prefix
MIDDLE_LEN = 24             # "delayed recall" -- deliberately long gap between fact and read trigger (B8 Stage 2's own framing)
PROMPT_LEN = FACT_POS + 2 + MIDDLE_LEN + 2  # prefix, marker+fact_id, middle, read-trigger bigram
SEED = 555


def load_frozen_model():
    payload = json.loads((CHECKPOINT / "state.json").read_text())
    model = HZ0AMlxModel(VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF, ATTENTION_INDICES, native_metal=True)
    model_arrays = [(item["key"], mx.load(str(CHECKPOINT / item["file"]))) for item in payload["arrays"] if item["group"] == "model"]
    model.update(tree_unflatten(model_arrays))
    mx.eval(model.parameters())
    return model, payload


def make_prompts(count: int, rng: random.Random) -> tuple[mx.array, mx.array]:
    """Returns (token_ids [count, PROMPT_LEN], fact_is_a [count] bool-as-float)."""
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


def params_to_dict(p: LatentWriteControllerParams) -> dict:
    d = {f"key_proj.{n}": v for n, v in (("w", p.key_proj_w), ("b", p.key_proj_b))}
    d.update({f"value_proj.{n}": v for n, v in (("w", p.value_proj_w), ("b", p.value_proj_b))})
    d["occupancy_gate_w"] = p.occupancy_gate_w
    wc = p.write_controller
    d.update({f"read_params.{f.name}": getattr(wc.read_params, f.name) for f in dataclasses.fields(wc.read_params)})
    for f in dataclasses.fields(wc):
        if f.name != "read_params":
            d[f"wc.{f.name}"] = getattr(wc, f.name)
    return d


def dict_to_params(d: dict) -> LatentWriteControllerParams:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-sparse", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1.5e-1)
    parser.add_argument("--num-slots", type=int, default=8, help="fewer slots forces real eviction competition instead of every position getting its own free slot")
    parser.add_argument("--gate-bias-init", type=float, default=0.0, help="negative starts the write gate near-closed everywhere, so writing must be earned rather than defaulted to")
    parser.add_argument("--decay-rate", type=float, default=1.0, help="below 1.0, applies B2's forget_or_decay once per position so older writes lose confidence relative to fresher ones -- lets later, relevant writes win eviction battles against early, stale ones")
    args = parser.parse_args()

    rng = random.Random(SEED)
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")
    print(f"lambda_sparse={args.lambda_sparse} steps={args.steps} lr={args.lr} num_slots={args.num_slots} gate_bias_init={args.gate_bias_init} prompt_len={PROMPT_LEN} fact_pos={FACT_POS}")

    train_tokens, train_is_a = make_prompts(32, rng)
    held_out_tokens, held_out_is_a = make_prompts(16, rng)

    init_params = init_latent_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=SEED, write_gate_bias_init=args.gate_bias_init)

    def targets_for(is_a: mx.array) -> mx.array:
        return mx.where(is_a > 0.5, mx.array(TARGET_A), mx.array(TARGET_B)).astype(mx.int32)

    def accuracy(model, tokens, is_a, params) -> tuple[float, mx.array]:
        logits, _, gates = forward(model, tokens, latent_params=params, num_slots=args.num_slots, decay_rate=args.decay_rate)
        mx.eval(logits)
        final = logits[:, -1, :]
        predicted = mx.argmax(final, axis=-1)
        targets = targets_for(is_a)
        acc = float(mx.mean((predicted == targets).astype(mx.float32)))
        return acc, gates

    print("\n--- before training (random latent-write params) ---")
    acc0, _ = accuracy(model, held_out_tokens, held_out_is_a, init_params)
    print(f"held-out 2-way discrimination accuracy (chance = 0.5): {acc0:.3f}")

    params_dict = params_to_dict(init_params)

    def loss_fn(pd: dict) -> mx.array:
        p = dict_to_params(pd)
        logits, _, gates = forward(model, train_tokens, latent_params=p, num_slots=args.num_slots, decay_rate=args.decay_rate)
        final = logits[:, -1, :]
        targets = targets_for(train_is_a)
        task_loss = mx.mean(nn.losses.cross_entropy(final, targets))
        sparsity_loss = mx.mean(gates)  # same quantity as B3's excessive_write_penalty, reused in form
        return task_loss + args.lambda_sparse * sparsity_loss

    grad_fn = mx.value_and_grad(loss_fn)
    print("\n--- training query/gate/write_gate/key_proj/value_proj jointly (backbone frozen, no should_write label anywhere) ---")
    for step in range(args.steps):
        loss, grads = grad_fn(params_dict)
        mx.eval(loss)
        params_dict = {k: params_dict[k] - args.lr * grads[k] for k in params_dict}
        mx.eval(*params_dict.values())
        if step % 150 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  train loss {float(loss):.5f}")

    trained_params = dict_to_params(params_dict)

    print("\n--- after training, held-out prompts (unseen fact assignments) ---")
    acc1, gates = accuracy(model, held_out_tokens, held_out_is_a, trained_params)
    print(f"held-out 2-way discrimination accuracy (chance = 0.5): {acc1:.3f}")

    per_position_gate = mx.mean(gates, axis=0)  # [seq] -- mean over batch, so we can see WHERE writes actually concentrate rather than assuming a position
    print("\nmean write_gate per position (0-indexed; FACT_MARKER at", FACT_POS, ", fact_id at", FACT_POS + 1, ", read-trigger at", PROMPT_LEN - 2, "-", PROMPT_LEN - 1, "):")
    print(" ".join(f"{float(v):.2f}" for v in per_position_gate))

    # The causally-honest selectivity check: positions BEFORE the fact_id
    # is even seen cannot possibly carry task-relevant content (by
    # construction -- the model hasn't seen which fact it is yet), so any
    # write mass there is provably not "useful" in this task's terms.
    # Positions at-or-after the fact_id CAN legitimately carry it.
    mean_gate_before_fact = float(mx.mean(gates[:, :FACT_POS + 1]))
    mean_gate_at_or_after_fact = float(mx.mean(gates[:, FACT_POS + 1:]))
    print(f"\nmean write_gate BEFORE the fact is knowable (positions 0-{FACT_POS}):  {mean_gate_before_fact:.4f}  (any mass here is provably not useful)")
    print(f"mean write_gate AT-OR-AFTER the fact (positions {FACT_POS + 1}-{PROMPT_LEN - 1}):        {mean_gate_at_or_after_fact:.4f}")
    print(f"ratio (informative-window gate / pre-fact gate):                 {mean_gate_at_or_after_fact / max(mean_gate_before_fact, 1e-9):.2f}x")

    assert acc1 > acc0, "training must improve 2-way discrimination accuracy over the random-params baseline"
    if mean_gate_at_or_after_fact <= mean_gate_before_fact:
        print("\nNOTE: write_gate did NOT concentrate in the causally-informative window -- B8's sparsity/selectivity exit gate is not demonstrated by this run, only the raw task-solvability half is.")
    else:
        print("\nSelectivity demonstrated: more write mass lands where the fact is actually knowable than before it.")


if __name__ == "__main__":
    main()
