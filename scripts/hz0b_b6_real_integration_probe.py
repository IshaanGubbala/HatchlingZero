"""HZ-0B B6, real integration: trains ONLY the read-only integration's own
projections (query/gate/value-to-hidden) against an oracle-populated memory
fact, with the real frozen HZ-0A hybrid checkpoint providing hidden states.

Backbone stays frozen throughout (`reference/hz0a_mlx_model.py` is never
updated, only read). Memory content itself is written via the oracle/
non-learned bypass (B1 decision 13) -- one fixed (key, value) pair,
never touched by gradient descent; only the LEARNED READ PATH (how a
hidden state turns into a query, how much to gate the readout in) is
trained. This stays inside B6's stated scope ("Do not allow writes yet")
-- B7 is what trains a write controller, not this.

What this answers, with real numbers rather than synthetic ones:
1. Can a trained read-only memory path make the frozen model predict a
   token it otherwise would not, when the right context is present?
   (a genuine memory-specific-task improvement, B6's exit gate)
2. Does that same trained memory path leave ordinary held-out text
   (which never touches the trigger context) alone? (materially-no-
   general-degradation, the other half of B6's exit gate, and the
   "unrelated memories do not corrupt output" check -- now with a
   TRAINED, not random, read path on real data)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from reference.hz0a_mlx_model import HZ0AMlxModel
from reference.hz0b_b6_hz0a_integration import forward
from reference.hz0b_memory_simulator import MemoryState, reset as memory_reset, write as memory_write
from reference.hz0b_readonly_integration import ReadOnlyIntegrationParams, init_readonly_integration

VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF = 24576, 768, 31, 12, 2304
ATTENTION_INDICES = (4, 9, 14, 19, 24, 29)
KEY_DIM = VALUE_DIM = 32
CHECKPOINT = Path("outputs/hz0a_stage2_100m_hybrid_seed7/native_metal_checkpoint_best_full_holdout")

TRIGGER_A, TRIGGER_B, TARGET = 19001, 19002, 19003  # fixed, arbitrary, avoid low ids (special tokens) and high ids near vocab edge
PROMPT_LEN = 16
NUM_TRAIN_PROMPTS, NUM_HELD_OUT_PROMPTS = 24, 8
SEED = 123


def load_frozen_model() -> HZ0AMlxModel:
    payload = json.loads((CHECKPOINT / "state.json").read_text())
    model = HZ0AMlxModel(VOCAB_SIZE, D_MODEL, LAYERS, HEADS, D_FF, ATTENTION_INDICES, native_metal=True)
    model_arrays = [(item["key"], mx.load(str(CHECKPOINT / item["file"]))) for item in payload["arrays"] if item["group"] == "model"]
    model.update(tree_unflatten(model_arrays))
    mx.eval(model.parameters())
    return model, payload


def make_probe_prompts(count: int, rng: random.Random) -> mx.array:
    """Random real-vocab prefixes, each ending in the fixed trigger bigram
    -- varying the prefix across prompts is what makes a positive result
    a genuine content-addressed generalization, not memorization of one
    exact sequence."""
    rows = []
    for _ in range(count):
        prefix = [rng.randint(100, VOCAB_SIZE - 100) for _ in range(PROMPT_LEN - 2)]
        rows.append(prefix + [TRIGGER_A, TRIGGER_B])
    return mx.array(rows, dtype=mx.int32)


def params_to_dict(p: ReadOnlyIntegrationParams) -> dict:
    return {f.name: getattr(p, f.name) for f in __import__("dataclasses").fields(p)}


def dict_to_params(d: dict) -> ReadOnlyIntegrationParams:
    return ReadOnlyIntegrationParams(**d)


def target_logit_stats(model, prompts: mx.array, memory_params, memory_state, *, confidence_scaled: bool = False) -> tuple[float, float, float]:
    """Returns (mean_target_logrank_no_memory, mean_target_logrank_with_memory, mean_ce_with_memory)
    at the FINAL position (predicting the token after the trigger bigram)."""
    logits_no_mem, _ = forward(model, prompts)
    logits_mem, _ = forward(model, prompts, memory_params=memory_params, memory_state=memory_state, confidence_scaled=confidence_scaled)
    mx.eval(logits_no_mem, logits_mem)
    final_no_mem = logits_no_mem[:, -1, :]
    final_mem = logits_mem[:, -1, :]

    def target_rank(logits_row) -> int:
        return int(mx.sum(logits_row > logits_row[TARGET]))

    ranks_no_mem = [target_rank(final_no_mem[i]) for i in range(final_no_mem.shape[0])]
    ranks_mem = [target_rank(final_mem[i]) for i in range(final_mem.shape[0])]
    ce_mem = float(mx.mean(nn.losses.cross_entropy(final_mem, mx.full((final_mem.shape[0],), TARGET, dtype=mx.int32))))
    return sum(ranks_no_mem) / len(ranks_no_mem), sum(ranks_mem) / len(ranks_mem), ce_mem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-preserve", type=float, default=0.0, help="weight on the background-text preservation loss; 0 reproduces the original (untuned) B6 probe result")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.5e-1)
    parser.add_argument("--confidence-scaled", action="store_true", help="structural fix: gate the read by retrieval confidence too, so a trained bias can't leak through on empty/irrelevant memory (see gated_memory_read's docstring)")
    args = parser.parse_args()

    rng = random.Random(SEED)
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")
    print(f"lambda_preserve={args.lambda_preserve} steps={args.steps} lr={args.lr} confidence_scaled={args.confidence_scaled}")

    train_prompts = make_probe_prompts(NUM_TRAIN_PROMPTS, rng)
    held_out_prompts = make_probe_prompts(NUM_HELD_OUT_PROMPTS, rng)

    # Background preservation set: a DIFFERENT slice of the same file than
    # the final degradation-eval slice below ([:64]), so training never
    # sees the sequences the reported held-out number is measured on --
    # otherwise "tuning against degradation" would be training on the eval.
    background_lines = Path("data/packed/repro_256_val.jsonl").open().readlines()[64:80]
    background_tokens = mx.array([json.loads(l)[:32] for l in background_lines], dtype=mx.int32)

    memory_key = mx.random.normal((1, KEY_DIM), key=mx.random.key(SEED))
    memory_value = mx.random.normal((1, VALUE_DIM), key=mx.random.key(SEED + 1))

    def oracle_memory(batch_size: int) -> MemoryState:
        base = memory_reset(batch_size=1, num_slots=8, key_dim=KEY_DIM, value_dim=VALUE_DIM)
        written, _, _ = memory_write(base, memory_key, memory_value, mx.array([1.0]), step=0, slot_idx=mx.array([0]))
        return MemoryState(**{
            f: mx.broadcast_to(getattr(written, f)[:1], (batch_size,) + getattr(written, f).shape[1:])
            for f in ("keys", "values", "confidence", "age", "protection", "write_count", "last_write_step", "write_source")
        })

    init_params = init_readonly_integration(D_MODEL, KEY_DIM, VALUE_DIM, seed=SEED)

    print("\n--- before training (random read-path params) ---")
    rank_no_mem, rank_mem_untrained, ce_untrained = target_logit_stats(model, held_out_prompts, init_params, oracle_memory(NUM_HELD_OUT_PROMPTS), confidence_scaled=args.confidence_scaled)
    print(f"mean target token rank, no memory:        {rank_no_mem:.1f} / {VOCAB_SIZE}")
    print(f"mean target token rank, untrained memory: {rank_mem_untrained:.1f} / {VOCAB_SIZE}  (expect: no better than no-memory, random query can't address the oracle key)")

    params_dict = params_to_dict(init_params)
    frozen_memory_train = oracle_memory(NUM_TRAIN_PROMPTS)
    frozen_memory_background = oracle_memory(background_tokens.shape[0])

    def loss_fn(pd: dict) -> mx.array:
        p = dict_to_params(pd)
        logits, _ = forward(model, train_prompts, memory_params=p, memory_state=frozen_memory_train, confidence_scaled=args.confidence_scaled)
        final_logits = logits[:, -1, :]
        targets = mx.full((final_logits.shape[0],), TARGET, dtype=mx.int32)
        task_loss = mx.mean(nn.losses.cross_entropy(final_logits, targets))
        if args.lambda_preserve == 0.0:
            return task_loss
        # Directly regularizes the exact metric B6's exit gate cares about
        # (next-token cross-entropy on real, trigger-free text with memory
        # populated) rather than a proxy -- penalizes the trained read
        # path for firing on content it has no business firing on.
        bg_logits, _ = forward(model, background_tokens, memory_params=p, memory_state=frozen_memory_background, confidence_scaled=args.confidence_scaled)
        preserve_loss = mx.mean(nn.losses.cross_entropy(bg_logits[:, :-1].astype(mx.float32), background_tokens[:, 1:]))
        return task_loss + args.lambda_preserve * preserve_loss

    grad_fn = mx.value_and_grad(loss_fn)
    print("\n--- training query/gate/value_to_hidden projections only (backbone frozen) ---")
    for step in range(args.steps):
        loss, grads = grad_fn(params_dict)
        mx.eval(loss)
        params_dict = {k: params_dict[k] - args.lr * grads[k] for k in params_dict}
        mx.eval(*params_dict.values())
        if step % 100 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  train loss {float(loss):.5f}")

    trained_params = dict_to_params(params_dict)

    print("\n--- after training, held-out prompts (unseen prefixes, same trigger) ---")
    rank_no_mem2, rank_mem_trained, ce_trained = target_logit_stats(model, held_out_prompts, trained_params, oracle_memory(NUM_HELD_OUT_PROMPTS), confidence_scaled=args.confidence_scaled)
    print(f"mean target token rank, no memory:        {rank_no_mem2:.1f} / {VOCAB_SIZE}")
    print(f"mean target token rank, TRAINED memory:   {rank_mem_trained:.1f} / {VOCAB_SIZE}")
    print(f"mean cross-entropy on target, trained memory: {ce_trained:.5f}")

    print("\n--- general held-out validation (real text, unrelated to the trigger fact), oracle memory still populated ---")
    val_lines = Path("data/packed/repro_256_val.jsonl").open().readlines()[:64]
    val_tokens = mx.array([json.loads(l)[:256] for l in val_lines], dtype=mx.int32)
    logits_no_mem, _ = forward(model, val_tokens)
    logits_trained_mem, _ = forward(model, val_tokens, memory_params=trained_params, memory_state=oracle_memory(val_tokens.shape[0]), confidence_scaled=args.confidence_scaled)
    mx.eval(logits_no_mem, logits_trained_mem)
    ce_val_no_mem = float(mx.mean(nn.losses.cross_entropy(logits_no_mem[:, :-1].astype(mx.float32), val_tokens[:, 1:])))
    ce_val_trained_mem = float(mx.mean(nn.losses.cross_entropy(logits_trained_mem[:, :-1].astype(mx.float32), val_tokens[:, 1:])))
    print(f"general held-out cross-entropy, no memory:            {ce_val_no_mem:.6f}")
    print(f"general held-out cross-entropy, trained memory (oracle-populated, no trigger present): {ce_val_trained_mem:.6f}")
    relative_change = (ce_val_trained_mem - ce_val_no_mem) / ce_val_no_mem * 100
    print(f"relative change: {relative_change:.3f}%")

    assert rank_mem_trained < rank_no_mem2, "trained memory must improve (lower) the target token's rank on held-out probe prompts"
    assert relative_change < 10.0, f"general held-out degradation ({relative_change:.2f}%) exceeds a generous 10% sanity bound -- investigate before trusting this result"


if __name__ == "__main__":
    main()
