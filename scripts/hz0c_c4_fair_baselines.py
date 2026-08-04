"""HZ-0C C4: fair anchor baselines. Per the plan: "Compare against no
anchors, fixed anchors, random anchors at matched rate, oracle
anchors, full attention, and an equal-compute transformer." Exit
gate: "quality can be compared at matched attention FLOPs."

Reuses C3's exact 8 real scenario constructions and ground-truth
positions (`scripts/hz0c_c3_trigger_simulator.py`) -- the real,
in-distribution content this project's own C2 lesson requires, not
synthetic random-ID tasks. Compares causal trigger policies at a MATCHED
~15% activation rate (except no-anchor at 0%, full-attention at 100%,
and oracle which is exactly the ground-truth rate): no anchors, fixed
periodic, random (matched rate), oracle, `state_novelty_score`,
`token_loss_score`, and causal next-token uncertainty.

The equal-compute transformer is a deterministic reference model with
the same hidden size/head geometry and six dense-attention layers (the
same six attention layers represented by HZ-0A's periodic schedule).
This matches attention FLOPs, not total training FLOPs. It is trained
as a causal LM on the same real corpus with fixed steps and seeds.
"""
from __future__ import annotations

import json
import random
import argparse

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from reference.hz0c_surprise_trigger import (
    fixed_periodic_trigger, full_attention_trigger, no_anchor_trigger, normalize_score, oracle_trigger,
    random_trigger, rate_bounded_threshold, state_novelty_score, token_loss_score,
)
from scripts.hz0b_b11_baseline_comparison import load_frozen_model
from reference.hz0b_b6_hz0a_integration import logits_from_hidden
from scripts.hz0c_c3_trigger_simulator import (
    CODE_DATA_PATH, GENERAL_DATA_PATH, JSON_DATA_PATH, SEQ_LEN, TARGET_RATE, load_real_sequences,
    scenario_code_json_boundary, scenario_contradiction, scenario_distractor_heavy_retrieval,
    scenario_long_range_reappearance, scenario_rare_token_burst, scenario_repeated_pattern_anomaly,
    scenario_topic_shift, scenario_variable_rebinding,
)
from reference.hz0b_b6_hz0a_integration import frozen_hidden_states


class EqualComputeTransformer(nn.Module):
    """Reference transformer with the periodic model's attention budget."""

    def __init__(self, vocab_size: int, dim: int, heads: int, d_ff: int, depth: int = 6):
        super().__init__()
        from reference.hz0a_mlx_model import Block

        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = [Block(dim, heads, d_ff, attention=True) for _ in range(depth)]
        self.final_norm = nn.RMSNorm(dim)
        self.depth = depth

    def __call__(self, token_ids):
        x = self.embedding(token_ids)
        for block in self.blocks:
            x, _ = block(x, None)
        return self.final_norm(x)


def train_equal_compute_transformer(transformer: EqualComputeTransformer, sequences: list[list[int]], steps: int = 256, learning_rate: float = 1e-4) -> list[float]:
    """Train the reference as a small causal LM on deterministic real data."""
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    batches = [mx.array(sequences[i:i + 4], dtype=mx.int32) for i in range(0, min(len(sequences), 128), 4)]

    def loss_fn(model, tokens):
        hidden = model(tokens)
        logits = mx.matmul(hidden, model.embedding.weight.T)
        return mx.mean(nn.losses.cross_entropy(logits[:, :-1].astype(mx.float32), tokens[:, 1:]))

    value_and_grad = nn.value_and_grad(transformer, loss_fn)
    losses = []
    for step in range(steps):
        tokens = batches[step % len(batches)]
        loss, grads = value_and_grad(transformer, tokens)
        optimizer.update(transformer, grads)
        mx.eval(transformer.parameters(), optimizer.state)
        losses.append(float(loss))
    return losses


def language_model_loss(model: EqualComputeTransformer, sequences: list[list[int]]) -> float:
    tokens = mx.array(sequences, dtype=mx.int32)
    hidden = model(tokens)
    logits = mx.matmul(hidden, model.embedding.weight.T)
    loss = mx.mean(nn.losses.cross_entropy(logits[:, :-1].astype(mx.float32), tokens[:, 1:]))
    mx.eval(loss)
    return float(loss)


def attention_flops(batch: int, seq_len: int, dim: int, heads: int, layers: int) -> int:
    """Leading-order QK^T/AV FLOPs for causal attention."""
    del heads  # head partitioning does not change the leading total.
    return 4 * batch * layers * seq_len * seq_len * dim


def transformer_parameter_count(model: nn.Module) -> int:
    return sum(value.size for _, value in tree_flatten(model.parameters()))


def fixed_matched_trigger(batch: int, seq: int, rate: float) -> mx.array:
    count = max(1, round(rate * seq))
    positions = mx.floor(mx.arange(count) * (seq / count)).astype(mx.int32)
    row = mx.zeros((seq,))
    row = row.at[positions].add(1.0)
    return mx.broadcast_to(row[None, :], (batch, seq))


def top_rate_trigger(score: mx.array, rate: float) -> mx.array:
    count = max(1, round(rate * score.shape[-1]))
    order = mx.argsort(-score, axis=-1)
    ranks = mx.argsort(order, axis=-1)
    return (ranks < count).astype(mx.float32)


def attention_demand_score(model, tokens: mx.array) -> mx.array:
    """Causal attention-demand score from the frozen anchor projections."""
    x = model.embedding(tokens)
    signals = []
    for index, block in enumerate(model.blocks):
        normed = block.norm1(x)
        if index in (4, 9, 14, 19, 24, 29):
            qkv = normed @ block.mixer.qkv.weight.T + block.mixer.qkv.bias
            q, k, v = mx.split(qkv, 3, axis=-1)
            signals.append(mx.stack([
                mx.mean(q * q, axis=-1), mx.mean(k * k, axis=-1),
                mx.mean(v * v, axis=-1), mx.var(qkv, axis=-1),
            ], axis=-1))
        x, _ = block(x, None)
    signal = mx.mean(mx.stack(signals, axis=0), axis=0)
    # Normalize each causal signal per sequence before combining them so one
    # projection scale cannot dominate the matched-rate ranking.
    signal = (signal - mx.mean(signal, axis=1, keepdims=True)) / (mx.std(signal, axis=1, keepdims=True) + 1e-5)
    return mx.mean(signal, axis=-1)


def layer_aware_demand_components(model, tokens: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Return component-relative causal mean/std/max demand summaries."""
    x = model.embedding(tokens)
    signals = []
    for index, block in enumerate(model.blocks):
        normed = block.norm1(x)
        if index in (4, 9, 14, 19, 24, 29):
            qkv = normed @ block.mixer.qkv.weight.T + block.mixer.qkv.bias
            q, k, v = mx.split(qkv, 3, axis=-1)
            signals.append(mx.stack([
                mx.mean(q * q, axis=-1), mx.mean(k * k, axis=-1),
                mx.mean(v * v, axis=-1), mx.var(qkv, axis=-1),
            ], axis=-1))
        x, _ = block(x, None)
    layer_demand = mx.stack(signals, axis=0)
    summaries = []
    for summary in (mx.mean(layer_demand, axis=0), mx.std(layer_demand, axis=0), mx.max(layer_demand, axis=0)):
        summary = (summary - mx.mean(summary, axis=1, keepdims=True)) / (
            mx.std(summary, axis=1, keepdims=True) + 1e-5
        )
        summaries.append(mx.mean(summary, axis=-1))
    relative = normalize_score(mx.stack(summaries, axis=-1))
    return relative[..., 0], relative[..., 1], relative[..., 2]


def causal_uncertainty_components(model, hidden: mx.array) -> mx.array:
    """Return causal entropy/confidence/margin components."""
    logits = logits_from_hidden(model, hidden)
    values = logits - mx.max(logits, axis=-1, keepdims=True)
    probabilities = mx.exp(values)
    probabilities = probabilities / mx.sum(probabilities, axis=-1, keepdims=True)
    entropy = -mx.sum(probabilities * mx.log(probabilities + 1e-8), axis=-1)
    top_two = mx.sort(probabilities, axis=-1)[..., -2:]
    confidence = -mx.log(top_two[..., -1] + 1e-8)
    margin = top_two[..., -1] - top_two[..., -2]
    signal = mx.stack([entropy, confidence, margin], axis=-1)
    signal = (signal - mx.mean(signal, axis=1, keepdims=True)) / (
        mx.std(signal, axis=1, keepdims=True) + 1e-5
    )
    return signal


def causal_uncertainty_score(model, hidden: mx.array) -> mx.array:
    """Rank positions by uncertainty available without the future token."""
    return mx.mean(causal_uncertainty_components(model, hidden), axis=-1)


def score_trigger(trigger: mx.array, gts: list[list[int]]) -> dict:
    seq_len = trigger.shape[1]
    tp = fp = fn = total_gt = total_positions = 0
    for i, gt_positions in enumerate(gts):
        gt_set = set(gt_positions)
        total_gt += len(gt_set)
        for t in range(seq_len):
            total_positions += 1
            is_triggered = bool(trigger[i, t] > 0.5)
            is_gt = t in gt_set
            if is_gt and is_triggered:
                tp += 1
            elif is_gt and not is_triggered:
                fn += 1
            elif not is_gt and is_triggered:
                fp += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    avg_rate = float(mx.mean(trigger))
    return {"precision": precision, "recall": recall, "avg_rate": avg_rate}


def evaluate_all_baselines(model, transformer, tokens: mx.array, gts: list[list[int]], seed: int) -> dict:
    hidden, _ = frozen_hidden_states(model, tokens)
    mx.eval(hidden)
    batch, seq_len = tokens.shape

    results = {}
    results["1. no_anchor"] = score_trigger(no_anchor_trigger(batch, seq_len), gts)
    results["2. fixed_periodic"] = score_trigger(fixed_matched_trigger(batch, seq_len, TARGET_RATE), gts)
    mx.random.seed(seed)
    random_scores = mx.random.uniform(shape=(batch, seq_len))
    results["3. random_matched"] = score_trigger(top_rate_trigger(random_scores, TARGET_RATE), gts)
    results["4. oracle"] = score_trigger(oracle_trigger(batch, seq_len, gts), gts)
    results["5. full_attention"] = score_trigger(full_attention_trigger(batch, seq_len), gts)

    state_novelty = normalize_score(state_novelty_score(hidden, window=4))
    state_novelty_trig = top_rate_trigger(state_novelty, TARGET_RATE)
    results["6. state_novelty (real-inference-safe)"] = score_trigger(state_novelty_trig, gts)

    demand = attention_demand_score(model, tokens)
    demand_trig = top_rate_trigger(demand, TARGET_RATE)
    results["6b. attention_demand (real-inference-safe)"] = score_trigger(demand_trig, gts)
    for alpha in (0.25, 0.5, 1.0, 2.0):
        blended = normalize_score(state_novelty + alpha * normalize_score(demand))
        results[f"6c. novelty+demand alpha={alpha:g}"] = score_trigger(
            top_rate_trigger(blended, TARGET_RATE), gts
        )

    uncertainty_components = causal_uncertainty_components(model, hidden)
    uncertainty = mx.mean(uncertainty_components, axis=-1)
    results["6d. causal_uncertainty (real-inference-safe)"] = score_trigger(
        top_rate_trigger(uncertainty, TARGET_RATE), gts
    )
    for component in range(3):
        component_score = uncertainty_components[..., component]
        results[f"6d{component + 1}. uncertainty_component_{component}"] = score_trigger(
            top_rate_trigger(component_score, TARGET_RATE), gts
        )
    entropy = uncertainty_components[..., 0]
    for weight in (0.25, 0.5, 1.0, 2.0):
        weighted = normalize_score(state_novelty + weight * normalize_score(entropy))
        results[f"6h. novelty+entropy weight={weight:g}"] = score_trigger(
            top_rate_trigger(weighted, TARGET_RATE), gts
        )
    for demand_weight in (0.0, 0.1, 0.25, 0.5, 1.0):
        for uncertainty_weight in (0.25, 0.5, 1.0, 2.0):
            blended = normalize_score(
                state_novelty
                + demand_weight * normalize_score(demand)
                + uncertainty_weight * normalize_score(uncertainty)
            )
            results[
                "6e. novelty+demand+uncertainty "
                f"demand={demand_weight:g} uncertainty={uncertainty_weight:g}"
            ] = score_trigger(top_rate_trigger(blended, TARGET_RATE), gts)
    for weights in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0), (1.0, 0.5, 0.5),
                    (0.5, 1.0, 0.5), (0.5, 0.5, 1.0)):
        weighted_uncertainty = mx.sum(
            uncertainty_components * mx.array(weights, dtype=mx.float32), axis=-1
        )
        blended = normalize_score(state_novelty + normalize_score(weighted_uncertainty))
        label = ",".join(f"{weight:g}" for weight in weights)
        results[f"6g. novelty+uncertainty_components weights={label}"] = score_trigger(
            top_rate_trigger(blended, TARGET_RATE), gts
        )
    # Keep the selected configuration explicit for downstream reports and
    # callers; the surrounding grid remains useful for auditing the choice.
    demand_mean, _, demand_max = layer_aware_demand_components(model, tokens)
    selected = normalize_score(
        state_novelty
        + normalize_score(uncertainty_components[..., 0])
        + 0.25 * normalize_score(demand_mean)
        + 0.5 * normalize_score(demand_max)
    )
    results["6f. selected novelty+entropy+layer-demand"] = score_trigger(
        top_rate_trigger(selected, TARGET_RATE), gts
    )
    # Test whether attention demand is more useful when it coincides with
    # causal uncertainty, rather than contributing as an independent score.
    entropy_norm = normalize_score(uncertainty_components[..., 0])
    demand_max_norm = normalize_score(demand_max)
    demand_uncertainty = normalize_score(demand_max_norm * entropy_norm)
    for interaction_weight in (0.1, 0.25, 0.5, 1.0):
        interaction_selected = normalize_score(
            selected + interaction_weight * demand_uncertainty
        )
        results[
            f"6i. selected+demand-uncertainty interaction={interaction_weight:g}"
        ] = score_trigger(
            top_rate_trigger(interaction_selected, TARGET_RATE), gts
        )

    token_loss = normalize_score(token_loss_score(model, hidden, tokens))
    token_loss_trig = top_rate_trigger(token_loss, TARGET_RATE)
    results["7. token_loss (offline-only)"] = score_trigger(token_loss_trig, gts)

    if transformer is not None:
        transformer_hidden = transformer(tokens)
        mx.eval(transformer_hidden)
        transformer_novelty = normalize_score(state_novelty_score(transformer_hidden, window=4))
        transformer_trig = top_rate_trigger(transformer_novelty, TARGET_RATE)
        results["8. equal_compute_transformer (trained)"] = score_trigger(transformer_trig, gts)

    return results


def main(training_steps: int = 256, learning_rate: float = 1e-4, skip_transformer: bool = False):
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}")

    general = load_real_sequences(GENERAL_DATA_PATH, 200)
    code = load_real_sequences(CODE_DATA_PATH, 100)
    json_seqs = load_real_sequences(JSON_DATA_PATH, 100)
    print(f"loaded {len(general)} general, {len(code)} code, {len(json_seqs)} json real sequences\n")

    transformer = None
    if not skip_transformer:
        mx.random.seed(0)
        transformer = EqualComputeTransformer(
            vocab_size=model.vocab_size, dim=model.dim, heads=model.heads,
            d_ff=model.blocks[0].up.weight.shape[0], depth=6,
        )
        train_losses = train_equal_compute_transformer(
            transformer, general, steps=training_steps, learning_rate=learning_rate
        )
        mx.eval(transformer.parameters())
        transformer_meta = {
            "depth": transformer.depth,
            "parameters": transformer_parameter_count(transformer),
            "attention_flops_per_example": attention_flops(1, SEQ_LEN, model.dim, model.heads, transformer.depth),
            "matched_hz0a_attention_layers": 6,
            "training_status": "trained_causal_lm_reference",
            "training_steps": training_steps,
            "train_loss_first": train_losses[0],
            "train_loss_last": train_losses[-1],
            "holdout_loss": language_model_loss(transformer, general[128:160]),
        }
        print("equal-compute transformer:", json.dumps(transformer_meta, sort_keys=True))

    rng = random.Random(555)
    NUM_EXAMPLES = 32

    scenarios = {
        "1. Repeated pattern with anomaly": scenario_repeated_pattern_anomaly(NUM_EXAMPLES, rng, general),
        "2. Topic shift": scenario_topic_shift(NUM_EXAMPLES, rng, general),
        "3. Long-range key reappearance": scenario_long_range_reappearance(NUM_EXAMPLES, rng, general),
        "4. Changed variable bindings": scenario_variable_rebinding(NUM_EXAMPLES, rng, general),
        "5. Code/JSON boundary": scenario_code_json_boundary(NUM_EXAMPLES, rng, code, json_seqs),
        "6. Contradiction": scenario_contradiction(NUM_EXAMPLES, rng, general),
        "7. Rare-token burst": scenario_rare_token_burst(NUM_EXAMPLES, rng, general, 24576),
        "8. Distractor-heavy retrieval": scenario_distractor_heavy_retrieval(NUM_EXAMPLES, rng, general, code),
    }

    aggregate = {}
    for name, (tokens, gts) in scenarios.items():
        results = evaluate_all_baselines(model, transformer, tokens, gts, seed=555)
        print(f"--- {name} ---")
        for policy, r in results.items():
            print(f"  {policy:38s}  precision={r['precision']:.3f}  recall={r['recall']:.3f}  rate={r['avg_rate']:.3f}")
            aggregate.setdefault(policy, []).append(r["recall"])
        print()

    print("--- Mean recall across all 8 scenarios, per baseline ---")
    for policy, recalls in aggregate.items():
        mean_recall = sum(recalls) / len(recalls)
        print(f"  {policy:38s}  mean_recall={mean_recall:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--skip-transformer", action="store_true")
    args = parser.parse_args()
    main(args.training_steps, args.learning_rate, args.skip_transformer)
