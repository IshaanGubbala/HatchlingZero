"""HZ-0D D8: curriculum.

Per the plan's own D8 text: "Progress through explicit update
supervision, few-shot rule inference, rule switching, natural temporary
preferences and schemas, and adversarial update/rollback tasks. Exit
gate: adaptation is sparse, quick, and reversible."

D6/D7 already built and verified the real-model wiring and the ordering
contract; D2/D3 already verified update supervision, few-shot inference,
rule interference, and adversarial/rollback safety in the ISOLATED
`dim=8` simulator. D8's real job is composing these into a curriculum
on the REAL frozen checkpoint and adding the one genuinely new piece:
a "natural schema" task whose INPUT distribution comes from real corpus
text through the real backbone, not synthetic Gaussian noise.

Honest operationalization note: this checkpoint is a pretrained,
NOT instruction-tuned language model -- there is no real "temporary
preference" (e.g. "reply in French") it could plausibly follow, and
building one for this checkpoint would be a strawman. The defensible,
disclosed substitution used here: keep D6's synthetic low-rank target
rule (a real, controlled adaptation signal), but draw the task's INPUT
(`x`) from the REAL attention-output activations the model produces on
REAL corpus text (`data/packed/repro_1024_val.jsonl`, the same file
`reference/hz0c_surprise_trigger.py`'s own scenarios use), rather than
synthetic Gaussian vectors. This makes the task "natural" in the one
sense that is actually meaningful for this checkpoint: its input
distribution, not an invented notion of instruction-following this
model was never trained for.
"""
from __future__ import annotations

import mlx.core as mx

from reference.hz0d_d6_integration import ATTENTION_INDICES
from reference.hz0d_isolated_simulator import Task


def collect_real_attention_output(model, token_ids: mx.array, layer_index: int, heads: int) -> mx.array:
    """Runs the REAL frozen backbone up to `layer_index`, then computes
    that layer's real causal self-attention output (post value-
    aggregation, PRE out-projection) -- exactly the `x` that
    `out_w`/`apply_fast_linear` consume, matching
    `reference/hz0c_surprise_trigger.py::masked_anchor_attention`'s own
    pre-out-projection computation, but un-triggered (every position
    attends normally) since this collects a natural REPRESENTATION, not
    a triggered one. Returns `[batch, seq, dim]`."""
    x = model.embedding(token_ids)
    for index, block in enumerate(model.blocks):
        if index == layer_index:
            normed = block.norm1(x)
            batch, seq, dim = normed.shape
            head_dim = dim // heads
            qkv = normed @ block.mixer.qkv.weight.T + block.mixer.qkv.bias
            q, k, v = mx.split(qkv.reshape(batch, seq, 3, heads, head_dim), 3, axis=2)
            q, k, v = (mx.squeeze(t, axis=2).transpose(0, 2, 1, 3) for t in (q, k, v))
            scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(head_dim, dtype=mx.float32))
            causal_mask = mx.triu(mx.full((seq, seq), -1e9), 1)
            weights = mx.softmax(scores + causal_mask[None, None], axis=-1)
            return mx.matmul(weights, v).transpose(0, 2, 1, 3).reshape(batch, seq, dim)
        x, _ = block(x, None)
    raise ValueError(f"layer_index {layer_index} not found among model.blocks (0..{len(model.blocks) - 1})")


def make_natural_schema_task(model, real_token_sequences: mx.array, *, layer_index: int = ATTENTION_INDICES[0], heads: int, seed: int, rule_scale: float = 0.05, k_train: int, k_held_out: int) -> Task:
    """A few-shot low-rank-remapping task (D2/D6's own shape:
    `true_delta = true_a @ true_b`, train/held-out split) whose `x`
    values are REAL attention-output activations collected from real
    corpus text (`collect_real_attention_output`), flattened across
    every `(batch, position)` pair, rather than synthetic Gaussian
    noise. `base_weight`/`base_bias` are the REAL frozen output
    projection at `layer_index`, matching D6's own convention."""
    layer = model.blocks[layer_index]
    real_w, real_b = layer.mixer.out.weight, layer.mixer.out.bias
    dim, rank = real_w.shape[0], 16

    attn_out = collect_real_attention_output(model, real_token_sequences, layer_index, heads)
    batch, seq, _ = attn_out.shape
    pool = attn_out.reshape(batch * seq, dim)
    needed = k_train + k_held_out
    if pool.shape[0] < needed:
        raise ValueError(f"need {needed} real activation vectors, only {pool.shape[0]} available -- pass more/longer real sequences")
    pool = pool[:needed]

    key = mx.random.key(seed)
    k_a, k_b = mx.random.split(key)
    true_a = mx.random.normal((dim, rank), key=k_a) * rule_scale
    true_b = mx.random.normal((rank, dim), key=k_b) * rule_scale
    true_delta = true_a @ true_b
    targets = pool @ (real_w + true_delta).T + real_b
    return Task(
        base_weight=real_w, base_bias=real_b, true_delta=true_delta,
        train_x=pool[:k_train], train_y=targets[:k_train],
        held_out_x=pool[k_train:], held_out_y=targets[k_train:],
    )
