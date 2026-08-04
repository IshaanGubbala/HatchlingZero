"""HZ-0C C6/C9: evaluate trigger policies in the frozen HZ-0A graph."""
from __future__ import annotations

import json
import random
import argparse

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from reference.hz0b_b6_hz0a_integration import frozen_hidden_states, logits_from_hidden
from reference.hz0b_b8_latent_write import init_latent_write_controller, sequential_latent_write_and_read
from reference.hz0c_surprise_trigger import (
    fixed_periodic_trigger, full_attention_trigger, masked_anchor_attention,
    no_anchor_trigger, normalize_score, random_trigger, rate_bounded_threshold,
    state_novelty_score,
    token_loss_score,
)
from scripts.hz0b_b11_baseline_comparison import load_frozen_model
from scripts.hz0c_c7_rl_trigger_controller import (
    bounded_actions, controller_features, fit_controller, fit_mlp_controller,
    mlp_logits,
)
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences

ATTENTION_INDICES = (4, 9, 14, 19, 24, 29)
TARGET_RATE = 0.15


def fixed_matched_trigger(batch: int, seq: int, rate: float) -> mx.array:
    count = max(1, round(rate * seq))
    positions = mx.floor(mx.arange(count) * (seq / count)).astype(mx.int32)
    row = mx.zeros((seq,))
    row = row.at[positions].add(1.0)
    return mx.broadcast_to(row[None, :], (batch, seq))


def exact_topk_labels(score: mx.array, rate: float) -> np.ndarray:
    count = max(1, round(rate * score.shape[-1]))
    order = np.argsort(-np.asarray(score), axis=-1)
    labels = np.zeros(order.shape, dtype=np.float32)
    rows = np.arange(order.shape[0])[:, None]
    labels[rows, order[:, :count]] = 1.0
    return labels


def conditional_hidden(model, token_ids: mx.array, trigger: mx.array) -> mx.array:
    """Replay HZ-0A with conditional attention at its six anchor layers,
    stopping BEFORE `final_norm` -- the same injection point
    `reference/hz0b_b6_hz0a_integration.py`/`hz0b_b8_latent_write.py`
    already use for HZ-0B memory (backbone residual stream fully formed,
    LM head not yet applied). Split out of the old inline
    `conditional_forward` specifically so memory can be inserted here
    without duplicating the anchor-attention loop."""
    x = model.embedding(token_ids)
    states = [None] * len(model.blocks)
    for index, (block, state) in enumerate(zip(model.blocks, states)):
        if index not in ATTENTION_INDICES:
            x, _ = block(x, state)
            continue
        normed = block.norm1(x)
        anchor = masked_anchor_attention(
            normed, trigger,
            qkv_w=block.mixer.qkv.weight, qkv_b=block.mixer.qkv.bias,
            out_w=block.mixer.out.weight, out_b=block.mixer.out.bias,
            heads=model.heads,
        )
        x = x + anchor
        normed2 = block.norm2(x)
        x = x + block.down(nn.silu(block.gate(normed2)) * block.up(normed2))
    return x


def conditional_forward(model, token_ids: mx.array, trigger: mx.array) -> mx.array:
    """Replay HZ-0A with conditional attention at its six anchor layers.
    Unchanged behavior from before the `conditional_hidden` split --
    every existing caller of this function sees identical output."""
    return logits_from_hidden(model, conditional_hidden(model, token_ids, trigger))


def conditional_forward_with_memory(model, token_ids: mx.array, trigger: mx.array, latent_params, *, decay_rate: float = 1.0, ste: bool = False) -> tuple[mx.array, mx.array]:
    """Wires HZ-0B's session-local write+read memory
    (`reference/hz0b_b8_latent_write.py::sequential_latent_write_and_read`)
    through the C6 trigger graph -- the open item the tracker names as
    "HZ-0B memory has not yet been wired into the trigger graph". Memory
    is inserted at the SAME point B6/B8 already use (after the
    conditional-attention backbone's residual stream is fully formed,
    before `final_norm`), so this is not a new integration convention,
    just C6's own graph reusing the existing one. Every position gets a
    real write+read pass (not B6's read-only oracle-populated bank),
    matching what B8-B11 actually mean by "HZ-0B memory". Returns
    `(logits, write_gates)` -- `write_gates`: [batch, seq], the same
    per-position sparsity-relevant quantity B8's own callers inspect."""
    hidden = conditional_hidden(model, token_ids, trigger)
    hidden, _, write_gates = sequential_latent_write_and_read(
        latent_params, hidden, decay_rate=decay_rate, ste=ste,
    )
    return logits_from_hidden(model, hidden), write_gates


def frozen_hidden_and_demand(model, token_ids: mx.array) -> tuple[mx.array, mx.array]:
    """Run the frozen backbone once and collect anchor demand in that pass.

    These features describe the actual attention projections that a trigger
    controls. They use only the current prefix state, not labels or future
    tokens, and avoid materializing an attention matrix.
    """
    x = model.embedding(token_ids)
    collected = []
    for index, block in enumerate(model.blocks):
        normed = block.norm1(x)
        if index in ATTENTION_INDICES:
            qkv = normed @ block.mixer.qkv.weight.T + block.mixer.qkv.bias
            q, k, v = mx.split(qkv, 3, axis=-1)
            collected.append(mx.stack([
                mx.mean(q * q, axis=-1),
                mx.mean(k * k, axis=-1),
                mx.mean(v * v, axis=-1),
                mx.var(qkv, axis=-1),
            ], axis=-1))
        x, _ = block(x, None)
    demand = mx.mean(mx.stack(collected, axis=0), axis=0)
    return x, demand


def attention_demand_features(model, token_ids: mx.array) -> mx.array:
    """Compatibility wrapper for callers that only need demand features."""
    _, demand = frozen_hidden_and_demand(model, token_ids)
    return demand


def controller_input(hidden: mx.array, demand: mx.array) -> np.ndarray:
    base = controller_features(hidden)
    extra = np.asarray(demand, dtype=np.float32)
    extra = (extra - extra.mean(1, keepdims=True)) / (extra.std(1, keepdims=True) + 1e-5)
    return np.concatenate([base, extra], axis=-1)


def loss_and_ppl(logits: mx.array, tokens: mx.array) -> tuple[float, float]:
    loss = mx.mean(nn.losses.cross_entropy(logits[:, :-1].astype(mx.float32), tokens[:, 1:]))
    mx.eval(loss)
    value = float(loss)
    return value, float(mx.exp(mx.array(value)))


def fit_downstream_controller(model, tokens: mx.array, features: np.ndarray, steps: int = 120, lr: float = 0.05) -> dict[str, mx.array]:
    """Optimize the actual frozen-backbone LM loss through a soft gate.

    The gate is soft only during training so gradients can reach the small
    controller. Evaluation still converts it to an exact top-15% policy.
    """
    feature_array = mx.array(features, dtype=mx.float32)
    params = {"w": mx.zeros((features.shape[-1],)), "b": mx.zeros((1,))}

    def objective(p):
        trigger = mx.sigmoid(feature_array @ p["w"] + p["b"])
        logits = conditional_forward(model, tokens, trigger)
        lm_loss = mx.mean(nn.losses.cross_entropy(
            logits[:, :-1].astype(mx.float32), tokens[:, 1:]
        ))
        rate_penalty = 8.0 * (mx.mean(trigger) - TARGET_RATE) ** 2
        return lm_loss + rate_penalty

    value_and_grad = mx.value_and_grad(objective)
    for _ in range(steps):
        loss, grads = value_and_grad(params)
        del loss
        params = {name: params[name] - lr * mx.clip(grads[name], -1.0, 1.0) for name in params}
        mx.eval(*params.values())
    return params


def causal_attention_benefit(model, tokens: mx.array, candidates: int = 32) -> np.ndarray:
    """Estimate each position's downstream loss benefit from one anchor.

    This is deliberately an offline teacher: it may use the frozen model and
    teacher-forced targets, but the resulting controller only receives causal
    hidden-state features at inference time. Keeping the teacher small makes
    its cost explicit rather than hiding it in the normal evaluation path.
    """
    batch, seq = tokens.shape
    baseline = mx.zeros((batch, seq))
    baseline_logits = conditional_forward(model, tokens, baseline)
    baseline_token_loss = nn.losses.cross_entropy(
        baseline_logits[:, :-1].astype(mx.float32), tokens[:, 1:], reduction="none"
    )
    mx.eval(baseline_token_loss)
    benefits = np.zeros((batch, seq), dtype=np.float32)
    # Restrict the expensive oracle to positions most likely to matter. The
    # cheap proxy is only used to choose candidates; labels still come from
    # the actual downstream loss delta.
    proxy = np.asarray(token_loss_score(model, frozen_hidden_states(model, tokens)[0], tokens))
    candidate_positions = [
        np.argsort(-proxy[row])[: min(seq, candidates)] for row in range(batch)
    ]
    for position in range(seq):
        active_rows = [row for row in range(batch) if position in candidate_positions[row]]
        if not active_rows:
            continue
        trigger = mx.zeros((batch, seq))
        trigger = trigger.at[active_rows, position].add(1.0)
        logits = conditional_forward(model, tokens, trigger)
        token_loss = nn.losses.cross_entropy(
            logits[:, :-1].astype(mx.float32), tokens[:, 1:], reduction="none"
        )
        mx.eval(token_loss)
        # A trigger can only affect its position and later positions. Ignore
        # the current token's prediction and score the causal suffix.
        if position + 1 < seq:
            values = np.asarray(
                baseline_token_loss[:, position:].astype(mx.float32) - token_loss[:, position:]
            ).mean(axis=1)
            for row in active_rows:
                benefits[row, position] = values[row]
    return benefits


def main(seed: int = 555, causal_teacher_sequences: int = 0, controller_kind: str = "linear", distill_steps: int = 1200, distill_lr: float = 0.2, positive_weight: float = 4.0, train_seeds: list[int] | None = None, causal_teacher_candidates: int = 32, causal_teacher_blend: float = 1.0, with_memory: bool = False, memory_seed: int = 17, memory_decay_rate: float = 1.0) -> None:
    model, _ = load_frozen_model()
    selected_train_seeds = train_seeds or [seed]
    train_hidden_parts, train_demand_parts, train_token_parts = [], [], []
    for train_seed in selected_train_seeds:
        train_corpus = load_real_sequences(GENERAL_DATA_PATH, 160)
        random.Random(train_seed).shuffle(train_corpus)
        train_tokens_part = mx.array(train_corpus[:128], dtype=mx.int32)
        hidden_part, demand_part = frozen_hidden_and_demand(model, train_tokens_part)
        mx.eval(hidden_part, demand_part)
        train_token_parts.append(train_tokens_part)
        train_hidden_parts.append(hidden_part)
        train_demand_parts.append(demand_part)
    train_tokens = mx.concatenate(train_token_parts, axis=0)
    train_hidden = mx.concatenate(train_hidden_parts, axis=0)
    train_demand = mx.concatenate(train_demand_parts, axis=0)
    if causal_teacher_sequences:
        teacher_tokens = train_tokens[:causal_teacher_sequences]
        teacher_benefit = causal_attention_benefit(model, teacher_tokens, candidates=causal_teacher_candidates)
        causal_score = normalize_score(mx.array(teacher_benefit))
        token_score = normalize_score(token_loss_score(model, train_hidden[:causal_teacher_sequences], teacher_tokens))
        blend = float(np.clip(causal_teacher_blend, 0.0, 1.0))
        teacher_score = (blend * causal_score) + ((1.0 - blend) * token_score)
        controller_features_train = controller_input(train_hidden[:causal_teacher_sequences], train_demand[:causal_teacher_sequences])
        teacher_labels = exact_topk_labels(teacher_score, TARGET_RATE)
    else:
        teacher_score = normalize_score(token_loss_score(model, train_hidden, train_tokens))
        controller_features_train = controller_input(train_hidden, train_demand)
        teacher_labels = exact_topk_labels(teacher_score, TARGET_RATE)
    if controller_kind == "downstream":
        # Use a small disjoint subset to keep the expensive graph-level
        # optimization bounded; the final policy is evaluated on all held-out
        # sequences below.
        downstream_params = fit_downstream_controller(
            model, train_tokens[:16], controller_features_train[:16], steps=120,
        )
        controller_params = downstream_params
    elif controller_kind == "mlp":
        controller_params = fit_mlp_controller(
            controller_features_train, teacher_labels, steps=1200,
        )
    else:
        controller_params = fit_controller(
            controller_features_train, teacher_labels, steps=distill_steps,
            lr=distill_lr, positive_weight=positive_weight,
        )
    corpus = load_real_sequences(GENERAL_DATA_PATH, 160)
    random.Random(seed).shuffle(corpus)
    sequences = corpus[128:160]
    tokens = mx.array(sequences, dtype=mx.int32)
    hidden, demand = frozen_hidden_and_demand(model, tokens)
    mx.eval(hidden, demand)
    scores = normalize_score(state_novelty_score(hidden, window=4))
    novelty = (scores > rate_bounded_threshold(scores, target_rate=TARGET_RATE, min_rate=0.02, max_rate=0.6)).astype(mx.float32)
    teacher_score = normalize_score(token_loss_score(model, hidden, tokens))
    token_loss_teacher = (teacher_score > rate_bounded_threshold(teacher_score, target_rate=TARGET_RATE, min_rate=0.02, max_rate=0.6)).astype(mx.float32)
    eval_features = controller_input(hidden, demand)
    if controller_kind == "downstream":
        controller_logits = np.asarray(
            eval_features @ controller_params["w"] + controller_params["b"], dtype=np.float32
        )
    elif controller_kind == "mlp":
        controller_logits = mlp_logits(eval_features, controller_params)
    else:
        controller_logits = eval_features @ controller_params[:-1] + controller_params[-1]
    controller_trigger = mx.array(bounded_actions(controller_logits, np.random.default_rng(seed), stochastic=False))
    policies = {
        "no_anchor": no_anchor_trigger(*tokens.shape),
        "fixed_periodic": fixed_matched_trigger(*tokens.shape, rate=TARGET_RATE),
        "random_matched": random_trigger(*tokens.shape, rate=TARGET_RATE, seed=555),
        "state_novelty": novelty,
        "token_loss_teacher_offline": token_loss_teacher,
        "learned_controller": controller_trigger,
        "full_attention": full_attention_trigger(*tokens.shape),
    }
    latent_params = None
    if with_memory:
        # Same key_dim/value_dim/num_slots convention already used by
        # `scripts/hz0c_c6_chunked_memory_audit.py` and B8-B11's own
        # protocol -- not a new choice, reusing the established default.
        # Untrained, fixed-seed params: this pass is about verifying the
        # trigger graph carries memory correctly and re-measuring matched-
        # cost LM loss with memory PRESENT, not about training a write
        # policy for general corpus text (a separate, larger undertaking
        # with no established B11-style protocol for this corpus).
        latent_params = init_latent_write_controller(model.dim, 32, 32, seed=memory_seed, write_gate_bias_init=-3.0)
    report = {}
    for name, trigger in policies.items():
        logits = conditional_forward(model, tokens, trigger)
        loss, ppl = loss_and_ppl(logits, tokens)
        entry = {"loss": loss, "perplexity": ppl, "anchor_rate": float(mx.mean(trigger))}
        if with_memory:
            memory_logits, write_gates = conditional_forward_with_memory(
                model, tokens, trigger, latent_params, decay_rate=memory_decay_rate,
            )
            memory_loss, memory_ppl = loss_and_ppl(memory_logits, tokens)
            entry["memory_loss"] = memory_loss
            entry["memory_perplexity"] = memory_ppl
            entry["memory_loss_delta"] = memory_loss - loss
            entry["mean_write_gate"] = float(mx.mean(write_gates))
        report[name] = entry
    print(json.dumps({"seed": seed, "train_seeds": selected_train_seeds, "attention_layers": len(ATTENTION_INDICES), "causal_teacher_sequences": causal_teacher_sequences, "controller_kind": controller_kind, "with_memory": with_memory, "memory_seed": memory_seed if with_memory else None, "memory_decay_rate": memory_decay_rate if with_memory else None, "policies": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--causal-teacher-sequences", type=int, default=0)
    parser.add_argument("--controller", choices=("linear", "mlp", "downstream"), default="linear")
    parser.add_argument("--causal-teacher-candidates", type=int, default=32,
                        help="Maximum candidate positions evaluated per teacher sequence.")
    parser.add_argument("--causal-teacher-blend", type=float, default=1.0,
                        help="Weight on causal downstream teacher versus token-loss teacher.")
    parser.add_argument("--distill-steps", type=int, default=1200)
    parser.add_argument("--distill-lr", type=float, default=0.2)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--with-memory", action="store_true",
                        help="Wire HZ-0B session-local write+read memory through the trigger graph and report matched-cost LM loss with memory present.")
    parser.add_argument("--memory-seed", type=int, default=17)
    parser.add_argument("--memory-decay-rate", type=float, default=1.0)
    args = parser.parse_args()
    main(args.seed, args.causal_teacher_sequences, args.controller,
         args.distill_steps, args.distill_lr, args.positive_weight,
         args.train_seeds, args.causal_teacher_candidates, args.causal_teacher_blend,
         args.with_memory, args.memory_seed, args.memory_decay_rate)
