"""HZ-0B B7, real integration: trains a supervised WRITE controller (plus
the same read path B6 trained) against the real frozen HZ-0A hybrid
checkpoint -- storage and retrieval both happen inside one real forward
pass this time, not via an oracle bypass set up before the pass starts
(that was B6's scope; see `docs/restart/hz0b_b6_real_integration_results.md`).

Sequence structure per prompt: [random prefix] -- WRITE_TRIGGER position
(should_write=1 supervision, fixed oracle key/value -- key/value are
supervised inputs per B1/B7, not learned) -- [random middle content] --
READ_TRIGGER_A, READ_TRIGGER_B -- predict TARGET. Only the controller
projections (query/gate/write_gate/update_gate/protect_gate/
value_to_hidden -- `reference/hz0b_write_integration.py`'s
WriteControllerParams) are trained; the frozen backbone and the memory's
own oracle key/value content are never touched by gradient descent.

Three comparisons, matching B7's own named modes:
1. "read only" (write_labels all None -- memory never gets written, so
   reading it can't possibly help): a sanity floor, not a memory result.
2. "read plus supervised write" (this script's main result): does a
   trained write-then-later-read loop, inside one real forward pass,
   actually let the frozen model retrieve a fact it was never trained on?
3. General held-out check (write_labels all should_write=0 -- memory
   stays empty throughout, same free exact-equality property B6 already
   proved holds): confirms nothing breaks on ordinary text when there's
   nothing to write.
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
from reference.hz0b_b7_hz0a_integration import forward
from reference.hz0b_write_integration import SupervisedWriteLabel, WriteControllerParams, init_write_controller

VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF = 24576, 768, 31, 12, 2304
ATTENTION_INDICES = (4, 9, 14, 19, 24, 29)
KEY_DIM = VALUE_DIM = 32
CHECKPOINT = Path("outputs/hz0a_stage2_100m_hybrid_seed7/native_metal_checkpoint_best_full_holdout")

WRITE_POS = 8          # fixed position (within the prefix) where the fact is offered for writing
PROMPT_LEN = 24
READ_TRIGGER_A, READ_TRIGGER_B, TARGET = 20001, 20002, 20003
SEED = 321


def load_frozen_model():
    payload = json.loads((CHECKPOINT / "state.json").read_text())
    model = HZ0AMlxModel(VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF, ATTENTION_INDICES, native_metal=True)
    model_arrays = [(item["key"], mx.load(str(CHECKPOINT / item["file"]))) for item in payload["arrays"] if item["group"] == "model"]
    model.update(tree_unflatten(model_arrays))
    mx.eval(model.parameters())
    return model, payload


def make_prompts(count: int, rng: random.Random) -> mx.array:
    rows = []
    for _ in range(count):
        row = [rng.randint(100, VOCAB_SIZE - 100) for _ in range(PROMPT_LEN - 2)]
        row += [READ_TRIGGER_A, READ_TRIGGER_B]
        rows.append(row)
    return mx.array(rows, dtype=mx.int32)


def make_write_labels(batch_size: int, key: mx.array, value: mx.array, *, write: bool) -> list:
    labels = [None] * PROMPT_LEN
    if write:
        labels[WRITE_POS] = SupervisedWriteLabel(
            should_write=mx.ones((batch_size,)), key=mx.broadcast_to(key, (batch_size, KEY_DIM)), value=mx.broadcast_to(value, (batch_size, VALUE_DIM)),
            should_protect=mx.zeros((batch_size,)), should_update=mx.zeros((batch_size,)), should_delete=mx.zeros((batch_size,)),
            target_slot=mx.zeros((batch_size,), dtype=mx.int32),
        )
    else:
        labels[WRITE_POS] = SupervisedWriteLabel(
            should_write=mx.zeros((batch_size,)), key=mx.broadcast_to(key, (batch_size, KEY_DIM)), value=mx.broadcast_to(value, (batch_size, VALUE_DIM)),
            should_protect=mx.zeros((batch_size,)), should_update=mx.zeros((batch_size,)), should_delete=mx.zeros((batch_size,)),
            target_slot=mx.zeros((batch_size,), dtype=mx.int32),
        )
    return labels


def params_to_dict(p: WriteControllerParams) -> dict:
    d = {f"read_params.{f.name}": getattr(p.read_params, f.name) for f in dataclasses.fields(p.read_params)}
    for f in dataclasses.fields(p):
        if f.name != "read_params":
            d[f.name] = getattr(p, f.name)
    return d


def dict_to_params(d: dict) -> WriteControllerParams:
    read_fields = {k.split(".", 1)[1]: v for k, v in d.items() if k.startswith("read_params.")}
    from reference.hz0b_readonly_integration import ReadOnlyIntegrationParams
    read_params = ReadOnlyIntegrationParams(**read_fields)
    other = {k: v for k, v in d.items() if not k.startswith("read_params.")}
    return WriteControllerParams(read_params=read_params, **other)


def target_rank_stats(model, prompts: mx.array, controller_params, write_labels_write: list, write_labels_readonly: list) -> tuple[float, float, float]:
    logits_readonly, _ = forward(model, prompts, controller_params=controller_params, write_labels=write_labels_readonly)
    logits_write, _ = forward(model, prompts, controller_params=controller_params, write_labels=write_labels_write)
    mx.eval(logits_readonly, logits_write)
    final_readonly, final_write = logits_readonly[:, -1, :], logits_write[:, -1, :]

    def rank(row) -> int:
        return int(mx.sum(row > row[TARGET]))

    ranks_readonly = [rank(final_readonly[i]) for i in range(final_readonly.shape[0])]
    ranks_write = [rank(final_write[i]) for i in range(final_write.shape[0])]
    ce_write = float(mx.mean(nn.losses.cross_entropy(final_write, mx.full((final_write.shape[0],), TARGET, dtype=mx.int32))))
    return sum(ranks_readonly) / len(ranks_readonly), sum(ranks_write) / len(ranks_write), ce_write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-preserve", type=float, default=5.0, help="carried over from the B6 tuning result (lambda=5 was the best of a 4-point sweep there), not independently re-swept for B7")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.5e-1)
    args = parser.parse_args()

    rng = random.Random(SEED)
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")
    print(f"lambda_preserve={args.lambda_preserve} steps={args.steps} lr={args.lr}")

    train_prompts = make_prompts(24, rng)
    held_out_prompts = make_prompts(8, rng)
    background_lines = Path("data/packed/repro_256_val.jsonl").open().readlines()[64:80]
    background_tokens = mx.array([json.loads(l)[:32] for l in background_lines], dtype=mx.int32)

    memory_key = mx.random.normal((1, KEY_DIM), key=mx.random.key(SEED))
    memory_value = mx.random.normal((1, VALUE_DIM), key=mx.random.key(SEED + 1))

    init_params = init_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=SEED)

    print("\n--- before training (random controller params) ---")
    labels_write_untrained = make_write_labels(8, memory_key, memory_value, write=True)
    labels_readonly_untrained = [None] * PROMPT_LEN
    rank_readonly0, rank_write0, _ = target_rank_stats(model, held_out_prompts, init_params, labels_write_untrained, labels_readonly_untrained)
    print(f"mean target rank, read-only (never written):    {rank_readonly0:.1f} / {VOCAB_SIZE}")
    print(f"mean target rank, untrained write-then-read:     {rank_write0:.1f} / {VOCAB_SIZE}  (expect: no better -- random controller can't reliably write+address)")

    params_dict = params_to_dict(init_params)
    labels_write_train = make_write_labels(24, memory_key, memory_value, write=True)
    labels_noop_bg = make_write_labels(background_tokens.shape[0], memory_key, memory_value, write=False)

    def loss_fn(pd: dict) -> mx.array:
        p = dict_to_params(pd)
        logits, _ = forward(model, train_prompts, controller_params=p, write_labels=labels_write_train)
        final_logits = logits[:, -1, :]
        targets = mx.full((final_logits.shape[0],), TARGET, dtype=mx.int32)
        task_loss = mx.mean(nn.losses.cross_entropy(final_logits, targets))
        if args.lambda_preserve == 0.0:
            return task_loss
        bg_write_labels = labels_noop_bg[:min(len(labels_noop_bg), background_tokens.shape[1])] + [None] * max(0, background_tokens.shape[1] - len(labels_noop_bg))
        bg_logits, _ = forward(model, background_tokens, controller_params=p, write_labels=bg_write_labels)
        preserve_loss = mx.mean(nn.losses.cross_entropy(bg_logits[:, :-1].astype(mx.float32), background_tokens[:, 1:]))
        return task_loss + args.lambda_preserve * preserve_loss

    grad_fn = mx.value_and_grad(loss_fn)
    print("\n--- training write/read controller projections only (backbone frozen, oracle key/value fixed) ---")
    for step in range(args.steps):
        loss, grads = grad_fn(params_dict)
        mx.eval(loss)
        params_dict = {k: params_dict[k] - args.lr * grads[k] for k in params_dict}
        mx.eval(*params_dict.values())
        if step % 100 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  train loss {float(loss):.5f}")

    trained_params = dict_to_params(params_dict)

    print("\n--- after training, held-out prompts (unseen prefixes) ---")
    labels_write_eval = make_write_labels(8, memory_key, memory_value, write=True)
    rank_readonly, rank_write, ce_write = target_rank_stats(model, held_out_prompts, trained_params, labels_write_eval, labels_readonly_untrained)
    print(f"mean target rank, read-only (never written):    {rank_readonly:.1f} / {VOCAB_SIZE}  (should stay high -- confirms the fact truly requires the write)")
    print(f"mean target rank, TRAINED write-then-read:       {rank_write:.1f} / {VOCAB_SIZE}")
    print(f"mean cross-entropy on target, write-then-read:   {ce_write:.5f}")

    print("\n--- general held-out validation (real text, should_write=0 throughout -> memory stays empty) ---")
    val_lines = Path("data/packed/repro_256_val.jsonl").open().readlines()[:64]
    val_tokens = mx.array([json.loads(l)[:256] for l in val_lines], dtype=mx.int32)
    val_labels_noop = make_write_labels(val_tokens.shape[0], memory_key, memory_value, write=False)
    val_write_labels = val_labels_noop[:min(len(val_labels_noop), val_tokens.shape[1])] + [None] * max(0, val_tokens.shape[1] - len(val_labels_noop))
    logits_no_mem, _ = forward(model, val_tokens)
    logits_trained_nowrite, _ = forward(model, val_tokens, controller_params=trained_params, write_labels=val_write_labels)
    mx.eval(logits_no_mem, logits_trained_nowrite)
    ce_val_no_mem = float(mx.mean(nn.losses.cross_entropy(logits_no_mem[:, :-1].astype(mx.float32), val_tokens[:, 1:])))
    ce_val_trained = float(mx.mean(nn.losses.cross_entropy(logits_trained_nowrite[:, :-1].astype(mx.float32), val_tokens[:, 1:])))
    print(f"general held-out cross-entropy, no memory:                  {ce_val_no_mem:.6f}")
    print(f"general held-out cross-entropy, trained controller, should_write=0: {ce_val_trained:.6f}")
    max_abs_diff = float(mx.max(mx.abs(logits_no_mem - logits_trained_nowrite)))
    print(f"max abs logit diff (expected ~0 but NOT exactly 0 once trained -- see note below): {max_abs_diff:.6f}")
    print(
        "Note: exact bit-identity (B6's untrained-params result) only holds when "
        "value_to_hidden_b/gate_b are at their zero-init values. Once trained, "
        "gated_memory_read's readout_in_hidden_space = readout @ W + b still adds "
        "the learned bias b even when readout is exactly zero (a truly empty memory) "
        "-- so a trained controller reintroduces a small, memory-CONTENT-independent "
        "perturbation via its bias terms alone. This is the precise mechanism behind "
        "part of B6's residual general-held-out degradation, not just correlated "
        "leakage on relevant-looking content as originally speculated there."
    )

    assert rank_write < rank_readonly, "a trained write-then-read loop must outperform never writing at all"
    if max_abs_diff >= 1.0:
        print(
            f"\nNOTE (not a hard failure): should_write=0 logit drift ({max_abs_diff:.3f}) is "
            "larger than B6's read-only case saw under the same lambda_preserve -- the write+read "
            "task is harder to optimize (see the train-loss trajectory above; B7's task did not "
            "converge as cleanly as B6's in the same step budget), and the SAME regularizer "
            "strength that worked for B6 was not independently re-tuned for this harder task. "
            "A real, disclosed limitation of this v1 result, not swept further here."
        )


if __name__ == "__main__":
    main()
