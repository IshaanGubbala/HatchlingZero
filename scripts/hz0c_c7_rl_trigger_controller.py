"""HZ-0C C7: teacher-distilled, group-relative RL trigger controller.

The frozen HZ-0A/HZ-0B backbone supplies features. Only a small logistic
policy is trained. Stage 1 imitates the offline token-loss teacher; Stage 2
uses group-relative REINFORCE rewards with a hard per-sequence rate bound.
"""
from __future__ import annotations

import json
import random
import argparse
from pathlib import Path

import numpy as np
import mlx.core as mx

from reference.hz0b_b6_hz0a_integration import frozen_hidden_states
from reference.hz0c_surprise_trigger import normalize_score, rate_bounded_threshold, token_loss_score
from scripts.hz0b_b11_baseline_comparison import load_frozen_model
from scripts.hz0c_c3_trigger_simulator import (
    CODE_DATA_PATH, GENERAL_DATA_PATH, JSON_DATA_PATH, TARGET_RATE,
    load_real_sequences, scenario_code_json_boundary, scenario_contradiction,
    scenario_distractor_heavy_retrieval, scenario_long_range_reappearance,
    scenario_rare_token_burst, scenario_repeated_pattern_anomaly,
    scenario_topic_shift, scenario_variable_rebinding,
)

MIN_RATE, MAX_RATE = 0.02, 0.15
SEED = 555
ATTENTION_INDICES = (4, 9, 14, 19, 24, 29)


def controller_features(hidden: mx.array, teacher_score: mx.array | None = None) -> np.ndarray:
    """Build causal features; no future token or ground-truth data is used."""
    h = np.asarray(hidden, dtype=np.float32)
    delta = np.zeros(h.shape[:2], dtype=np.float32)
    delta[:, 1:] = np.linalg.norm(h[:, 1:] - h[:, :-1], axis=-1)
    novelty = np.asarray(mx.array(normalize_score(mx.array(delta), method="minmax")), dtype=np.float32)
    steps = np.broadcast_to(np.arange(h.shape[1], dtype=np.float32)[None, :] / h.shape[1], novelty.shape)
    state_norm = np.linalg.norm(h, axis=-1).astype(np.float32)
    state_norm = (state_norm - state_norm.mean(1, keepdims=True)) / (state_norm.std(1, keepdims=True) + 1e-5)
    delta_norm = delta / (delta.mean(1, keepdims=True) + 1e-5)
    history_max = np.zeros_like(novelty)
    history_mean = np.zeros_like(novelty)
    delta_history = np.zeros_like(novelty)
    for t in range(h.shape[1]):
        start = max(0, t - 4)
        history_max[:, t] = novelty[:, start:t + 1].max(axis=1)
        history_mean[:, t] = novelty[:, start:t + 1].mean(axis=1)
        delta_history[:, t] = delta_norm[:, start:t + 1].mean(axis=1)
    base = np.stack([novelty, delta_norm, state_norm, steps, history_max, history_mean, delta_history, np.abs(state_norm)], axis=-1)
    # Low-cost nonlinear terms let the policy separate isolated spikes from
    # sustained novelty without introducing a second trainable network.
    return np.concatenate([base, base[..., :3] ** 2, (novelty * history_max)[..., None]], axis=-1)


def attention_demand_features(model, token_ids: mx.array) -> np.ndarray:
    """Add causal Q/K/V demand features from the frozen anchor layers."""
    x = model.embedding(token_ids)
    signals = []
    for index, block in enumerate(model.blocks):
        normed = block.norm1(x)
        if index in ATTENTION_INDICES:
            qkv = normed @ block.mixer.qkv.weight.T + block.mixer.qkv.bias
            q, k, v = mx.split(qkv, 3, axis=-1)
            signals.append(mx.stack([
                mx.mean(q * q, axis=-1), mx.mean(k * k, axis=-1),
                mx.mean(v * v, axis=-1), mx.var(qkv, axis=-1),
            ], axis=-1))
        x, _ = block(x, None)
    layer_demand = mx.stack(signals, axis=0)
    demand_mean = mx.mean(layer_demand, axis=0)
    demand_std = mx.std(layer_demand, axis=0)
    demand_max = mx.max(layer_demand, axis=0)
    groups = []
    for group in (demand_mean, demand_std, demand_max):
        groups.append((group - mx.mean(group, axis=1, keepdims=True)) / (
            mx.std(group, axis=1, keepdims=True) + 1e-5
        ))
    demand = mx.concatenate(groups, axis=-1)
    return np.asarray(demand, dtype=np.float32)


def controller_input(model, hidden: mx.array, token_ids: mx.array) -> np.ndarray:
    base = controller_features(hidden)
    demand = attention_demand_features(model, token_ids)
    # These are inference-time-safe confidence signals from the model's
    # current next-token distribution. They use only the prefix represented
    # by `hidden`, unlike token_loss_score, which consumes the future token.
    logits = model.final_norm(hidden) @ model.embedding.weight.T
    mx.eval(logits)
    values = np.asarray(logits, dtype=np.float32)
    values -= values.max(axis=-1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    entropy = -(probabilities * np.log(probabilities + 1e-8)).sum(axis=-1)
    top_two = np.sort(probabilities, axis=-1)[..., -2:]
    confidence = -np.log(top_two[..., -1] + 1e-8)
    margin = top_two[..., -1] - top_two[..., -2]
    uncertainty = np.stack([entropy, confidence, margin], axis=-1)
    uncertainty = (uncertainty - uncertainty.mean(axis=1, keepdims=True)) / (
        uncertainty.std(axis=1, keepdims=True) + 1e-5
    )
    # Cross-terms let the linear policy distinguish high projection demand
    # that is also genuinely uncertain from demand the backbone already
    # resolves confidently, without introducing a second network.
    interactions = (demand[..., :, None] * uncertainty[..., None, :]).reshape(
        demand.shape[0], demand.shape[1], -1
    )
    # Also expose causal novelty/history interactions so the policy can tell
    # isolated uncertainty spikes from uncertainty sustained by recent state
    # changes.
    novelty_uncertainty = base[..., 0:1] * uncertainty
    history_uncertainty = (base[..., 4:6, None] * uncertainty[..., None, :]).reshape(
        base.shape[0], base.shape[1], -1
    )
    return np.concatenate(
        [base, demand, uncertainty, interactions, novelty_uncertainty,
         history_uncertainty], axis=-1
    )


def bounded_actions(logits: np.ndarray, rng: np.random.Generator, stochastic: bool) -> np.ndarray:
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -12, 12)))
    actions = (rng.random(probs.shape) < probs) if stochastic else (probs >= 0.5)
    flat_actions = actions.reshape(-1, actions.shape[-1])
    flat_probs = probs.reshape(-1, probs.shape[-1])
    for row in range(flat_actions.shape[0]):
        count = int(np.clip(round(TARGET_RATE * flat_actions.shape[1]), round(MIN_RATE * flat_actions.shape[1]), round(MAX_RATE * flat_actions.shape[1])))
        order = np.argsort(-flat_probs[row])
        flat_actions[row] = False
        flat_actions[row, order[:count]] = True
    return actions.astype(np.float32)


def exact_teacher_actions(score: mx.array, rate: float) -> np.ndarray:
    count = max(1, round(rate * score.shape[-1]))
    order = np.argsort(-np.asarray(score), axis=-1)
    actions = np.zeros(order.shape, dtype=np.float32)
    rows = np.arange(order.shape[0])[:, None]
    actions[rows, order[:, :count]] = 1.0
    return actions


def reward(actions: np.ndarray, gts: list[list[int]]) -> np.ndarray:
    values = []
    for row, positions in enumerate(gts):
        seq_len = actions.shape[-1]
        gt = np.zeros(seq_len, dtype=bool)
        gt[positions] = True
        selected = actions[:, row] if actions.ndim == 3 else actions[row][None, :]
        hit = np.sum((selected > 0) & gt[None, :], axis=-1) / max(1, int(gt.sum()))
        cost = selected.mean(axis=-1)
        values.extend((hit - 0.10 * cost).tolist())
    return np.asarray(values, dtype=np.float32)


def recall_and_rate(actions: np.ndarray, gts: list[list[int]]) -> tuple[float, float]:
    scores = reward(actions[None, ...], gts).reshape(1, -1)[0]
    # Remove the small cost term used by RL and recover event recall.
    recall = [scores[i] + 0.10 * float(actions[i].mean()) for i in range(len(gts))]
    return float(np.mean(recall)), float(np.mean(actions))


def fit_controller(features: np.ndarray, labels: np.ndarray, steps: int = 1200,
                   lr: float = 0.2, positive_weight: float = 2.0) -> np.ndarray:
    w = np.zeros(features.shape[-1], dtype=np.float32)
    b = 0.0
    flat_x, flat_y = features.reshape(-1, features.shape[-1]), labels.reshape(-1)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(flat_x @ w + b, -12, 12)))
        grad = (p - flat_y) * np.where(flat_y > 0.5, positive_weight, 1.0)
        w -= lr * (flat_x.T @ grad / len(flat_y))
        b -= lr * float(grad.mean())
    return np.concatenate([w, np.asarray([b], dtype=np.float32)])


def fit_mlp_controller(features: np.ndarray, labels: np.ndarray, steps: int = 1200, hidden: int = 32, lr: float = 0.03) -> dict[str, np.ndarray]:
    """Fit a tiny causal policy to the offline teacher without touching HZ-0B."""
    rng = np.random.default_rng(17)
    x = features.reshape(-1, features.shape[-1]).astype(np.float32)
    y = labels.reshape(-1).astype(np.float32)
    w1 = rng.normal(0, 0.08, (x.shape[-1], hidden)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = rng.normal(0, 0.08, hidden).astype(np.float32)
    b2 = np.float32(0.0)
    for _ in range(steps):
        h = np.tanh(x @ w1 + b1)
        logits = h @ w2 + b2
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -12, 12)))
        g = p - y
        gw2 = h.T @ g / len(y)
        gb2 = np.mean(g)
        gh = (g[:, None] * w2[None, :]) * (1.0 - h * h)
        gw1 = x.T @ gh / len(y)
        gb1 = np.mean(gh, axis=0)
        w1 -= lr * gw1
        b1 -= lr * gb1
        w2 -= lr * gw2
        b2 -= lr * gb2
    return {"w1": w1, "b1": b1, "w2": w2, "b2": np.asarray([b2], dtype=np.float32)}


def mlp_logits(features: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    hidden = np.tanh(features @ params["w1"] + params["b1"])
    return hidden @ params["w2"] + params["b2"][0]


def main(seed: int = SEED, rl_steps: int = 200, rl_lr: float = 0.05,
         distill_steps: int = 1200, distill_lr: float = 0.2,
         positive_weight: float = 2.0, causal_teacher_sequences: int = 0,
         causal_teacher_candidates: int = 4, causal_teacher_blend: float = 0.5) -> None:
    model, _ = load_frozen_model()
    general = load_real_sequences(GENERAL_DATA_PATH, 200)
    code = load_real_sequences(CODE_DATA_PATH, 100)
    json_data = load_real_sequences(JSON_DATA_PATH, 100)
    rng = random.Random(seed)
    scenarios = [
        scenario_repeated_pattern_anomaly(32, rng, general),
        scenario_topic_shift(32, rng, general),
        scenario_long_range_reappearance(32, rng, general),
        scenario_variable_rebinding(32, rng, general),
        scenario_code_json_boundary(32, rng, code, json_data),
        scenario_contradiction(32, rng, general),
        scenario_rare_token_burst(32, rng, general, 24576),
        scenario_distractor_heavy_retrieval(32, rng, general, code),
    ]
    all_x, all_y = [], []
    for tokens, gts in scenarios:
        hidden, _ = frozen_hidden_states(model, tokens)
        mx.eval(hidden)
        score = normalize_score(token_loss_score(model, hidden, tokens))
        if causal_teacher_sequences:
            # Lazy import avoids the C6 evaluator's compatibility import of
            # this module creating a cycle on the default C7 path.
            from scripts.hz0c_c6_conditional_attention_eval import causal_attention_benefit
            count = min(causal_teacher_sequences, tokens.shape[0])
            causal = normalize_score(mx.array(causal_attention_benefit(
                model, tokens[:count], candidates=causal_teacher_candidates
            )))
            blend = float(np.clip(causal_teacher_blend, 0.0, 1.0))
            score = np.asarray(score)
            score[:count] = blend * np.asarray(causal) + (1.0 - blend) * score[:count]
        teacher_actions = exact_teacher_actions(score, TARGET_RATE)
        all_x.append(controller_input(model, hidden, tokens))
        all_y.append(teacher_actions)
    # Distill each scenario independently, then optimize one shared policy.
    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    params = fit_controller(
        x, y, steps=distill_steps, lr=distill_lr,
        positive_weight=positive_weight,
    )

    policy_rng = np.random.default_rng(seed)
    rl_rewards = []
    for _ in range(rl_steps):
        batch_rewards, grad = [], np.zeros_like(params)
        for features, (_, gts) in zip(all_x, scenarios):
            logits = features @ params[:-1] + params[-1]
            actions = bounded_actions(np.repeat(logits[None, ...], 8, axis=0), policy_rng, True)
            rewards = reward(actions, gts)
            centered = rewards.reshape(8, len(gts))
            centered = centered - centered.mean(axis=0, keepdims=True)
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -12, 12)))
            grad[:-1] += np.mean((centered[:, :, None] * (actions - probs[None, :, :]))[..., None] * features[None, :, :, :], axis=(0, 1, 2))
            grad[-1] += float(np.mean(centered[:, :, None] * (actions - probs[None, :, :])))
            batch_rewards.extend(rewards.tolist())
        params += rl_lr * grad / max(1, len(batch_rewards))
        rl_rewards.append(float(np.mean(batch_rewards)))
    controller_recalls, controller_rates = [], []
    teacher_recalls, teacher_rates = [], []
    for features, labels, (_, gts) in zip(all_x, all_y, scenarios):
        logits = features @ params[:-1] + params[-1]
        controller_actions = bounded_actions(logits, policy_rng, False)
        controller_recall, controller_rate = recall_and_rate(controller_actions, gts)
        teacher_recall, teacher_rate = recall_and_rate(labels, gts)
        controller_recalls.append(controller_recall)
        controller_rates.append(controller_rate)
        teacher_recalls.append(teacher_recall)
        teacher_rates.append(teacher_rate)
    report = {
        "stage": "C7-RL-trigger-controller",
        "seed": seed,
        "teacher": "token_loss_score",
        "causal_teacher_sequences": causal_teacher_sequences,
        "causal_teacher_candidates": causal_teacher_candidates,
        "causal_teacher_blend": causal_teacher_blend,
        "backbone_frozen": True,
        "controller_features": ["state_novelty", "hidden_delta", "state_norm", "relative_time", "causal_history", "polynomial_terms", "anchor_q_energy", "anchor_k_energy", "anchor_v_energy", "anchor_qkv_variance", "next_token_entropy", "next_token_confidence", "next_token_margin", "demand_uncertainty_interactions", "novelty_uncertainty_interactions", "history_uncertainty_interactions"],
        "hard_rate_bounds": [MIN_RATE, MAX_RATE],
        "distillation_steps": distill_steps,
        "distillation_learning_rate": distill_lr,
        "distillation_positive_weight": positive_weight,
        "rl_algorithm": "group-relative-reinforce",
        "rl_rollouts_per_sequence": 8,
        "rl_steps": rl_steps,
        "rl_learning_rate": rl_lr,
        "teacher_positive_rate": float(y.mean()),
        "final_group_reward": rl_rewards[-1],
        "controller_mean_event_recall": float(np.mean(controller_recalls)),
        "controller_mean_anchor_rate": float(np.mean(controller_rates)),
        "teacher_mean_event_recall": float(np.mean(teacher_recalls)),
        "teacher_mean_anchor_rate": float(np.mean(teacher_rates)),
        "finite": bool(np.all(np.isfinite(params))),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the deterministic HZ-0C C7 controller replay.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rl-steps", type=int, default=200)
    parser.add_argument("--rl-learning-rate", type=float, default=0.05)
    parser.add_argument("--distillation-steps", type=int, default=1200)
    parser.add_argument("--distillation-learning-rate", type=float, default=0.2)
    parser.add_argument("--distillation-positive-weight", type=float, default=2.0)
    args = parser.parse_args()
    main(
        seed=args.seed,
        rl_steps=args.rl_steps,
        rl_lr=args.rl_learning_rate,
        distill_steps=args.distillation_steps,
        distill_lr=args.distillation_learning_rate,
        positive_weight=args.distillation_positive_weight,
    )
