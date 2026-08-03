"""HZ-0C C1/C2: surprise-triggered anchor attention -- a real, minimal,
CORRECTNESS-FIRST reference implementation (matching this project's
established discipline of a slow, obviously-correct reference before
any optimized/bounded kernel -- HZ-0B's B2 simulator was built the
same way before PMetal). Per `plans/HZ-0C_Surprise_Anchors_Total_Restart_Plan.md`:

> Can HZ spend quadratic attention only when the recurrent state
> encounters something unexpected, preserving or improving quality at
> lower average attention cost?

`SurpriseTriggeredBlock` gives every position a GDN2 recurrent pass
(matching HZ-0A's existing recurrent layers exactly), THEN computes a
per-position surprise score and trigger decision, THEN adds a bounded
attention contribution gated by that trigger: only TRIGGERED positions
query, and only TRIGGERED positions serve as keys/values -- attention
is restricted to the "anchor" set via masking (not yet via a smaller
gathered KV cache; that optimization is C8's job, this is the
reference it must match). Non-triggered positions get zero attention
contribution at this layer (pure recurrent passthrough).

Surprise signal (C2's simplest candidate, chosen for zero additional
learned parameters beyond a tiny calibration term): hidden-state delta
norm, `||x_t - x_{t-1}||` per position, since the recurrent state's own
per-step change is a direct, cheap proxy for "how much did processing
this token move the state" -- no teacher-forced next-token loss needed
(which would require access unavailable at real inference time), no
new HZ-0B integration needed (deferred, real future C2 candidate:
memory-read uncertainty, once C6 wires this to frozen HZ-0B).

**C2 validation finding (2026-08-02, `docs/restart/hz0c_c2_surprise_validation_results.md`),
disclosed here so it isn't missed**: against the real frozen
checkpoint, this signal correlates well with general DIFFICULTY
(random vs. constant tokens: 7.09x higher mean surprise, correct
direction) but FAILS at novelty-POINT detection specifically -- a
single anomalous token injected into an otherwise-repeating pattern
scores LOWER than ordinary steady-state positions (wrong direction,
0% trigger rate on 32 injected anomalies at a 15%-target-rate
threshold). Usable as a difficulty/entropy proxy; NOT yet validated
as a novelty-point trigger signal for C3's simulator without further
work (a different C2 candidate, or a fix to this one).

**Follow-up (same date, same doc)**: `state_novelty_score()` (below)
was built specifically to fix this and ALSO fails on the same
construction (window=4 is worse than delta-norm; window=8 marginally
less bad but still wrong-direction) despite working correctly on a
synthetic toy pattern-break test. Two mechanistically different
signals failing the same way on the same real-model construction
points toward a TASK-CONSTRUCTION issue (random, semantically
meaningless token-ID cycles may not form a real "expectation" for a
language-trained model to violate) rather than two independently bad
signal choices.

**Hypothesis CONFIRMED (same date, `scripts/hz0c_c2_natural_novelty_validation.py`)**:
rebuilt with a real n-gram from real packed training data and a real,
in-distribution anomaly token -- both signals flip to the correct
direction. `state_novelty_score` (window 4-8) works well: 90.6% of
examples score the anomaly above steady-state (vs. 3-9% on the
random-ID construction), 47-53% of novelty positions trigger at a
15%-target rate (vs. 0%). `surprise_score` also flips but far more
weakly. C2's exit gate is genuinely met via `state_novelty_score` on
real, in-distribution content -- the random-ID confound is now a
documented lesson for C3's own task construction (use real,
in-distribution content, not arbitrary token IDs).
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from reference.hz0a_mlx_model import GDN2, Block


def surprise_score(hidden: mx.array) -> mx.array:
    """hidden: [batch, seq, dim] -> [batch, seq]. Position 0 has no
    prior state to compare against -- scored 0 (never triggers on its
    own, matching "nothing unexpected about the very first token")."""
    batch, seq, dim = hidden.shape
    delta = hidden[:, 1:, :] - hidden[:, :-1, :]
    delta_norm = mx.sqrt(mx.sum(delta * delta, axis=-1) + 1e-8)
    first = mx.zeros((batch, 1))
    return mx.concatenate([first, delta_norm], axis=1)


def state_novelty_score(hidden: mx.array, *, window: int = 4, eps: float = 1e-6) -> mx.array:
    """C2's "state novelty" candidate -- built specifically because
    `surprise_score` (hidden-state delta norm, i.e. novelty relative
    ONLY to the immediately preceding position) was validated
    (`docs/restart/hz0c_c2_surprise_validation_results.md`) to FAIL at
    detecting a single anomalous token in an otherwise-repeating
    pattern: a recurrent state that integrates gradually may not show
    a large token-to-token jump even at a genuine anomaly, since the
    anomaly's effect can be smoothed into the ongoing state rather
    than spiking it. This measures novelty relative to a RECENT WINDOW
    instead of just one prior position: cosine distance between the
    current hidden state and the causal mean of the previous `window`
    positions (no future leakage) -- directly targets "does this
    position look different from what's been typical recently," which
    is what a pattern-breaking anomaly should trigger even when it
    doesn't move the state much step-to-step.

    hidden: [batch, seq, dim] -> [batch, seq]. The first position has
    no window to compare against -- scored 0, matching `surprise_score`'s
    own convention."""
    batch, seq, dim = hidden.shape
    scores = [mx.zeros((batch,))]
    for t in range(1, seq):
        start = max(0, t - window)
        recent_mean = mx.mean(hidden[:, start:t, :], axis=1)
        current = hidden[:, t, :]
        cos_sim = mx.sum(current * recent_mean, axis=-1) / (
            mx.sqrt(mx.sum(current * current, axis=-1) + eps) * mx.sqrt(mx.sum(recent_mean * recent_mean, axis=-1) + eps)
        )
        scores.append(1.0 - cos_sim)
    return mx.stack(scores, axis=1)


def normalize_score(score: mx.array, *, method: str = "zscore", eps: float = 1e-6) -> mx.array:
    """C2's own required spec item: "specify normalization." score:
    [batch, seq]. Raw `surprise_score` magnitude depends on `dim` and
    training dynamics, not directly comparable across sequences or
    checkpoints -- normalize PER SEQUENCE (each row of the batch
    independently, not across the batch) so a threshold set once is
    meaningful regardless of a particular sequence's raw scale.

    `"zscore"` (default): `(score - mean) / (std + eps)` per sequence.
    `"minmax"`: rescale each sequence's own [min, max] to [0, 1].
    Position 0 is always exactly 0 in raw `surprise_score` by
    construction -- included in the normalization statistics like any
    other position (not special-cased), since excluding it would bias
    the mean/std for short sequences."""
    if method == "zscore":
        mean = mx.mean(score, axis=-1, keepdims=True)
        std = mx.std(score, axis=-1, keepdims=True)
        return (score - mean) / (std + eps)
    if method == "minmax":
        lo = mx.min(score, axis=-1, keepdims=True)
        hi = mx.max(score, axis=-1, keepdims=True)
        return (score - lo) / (hi - lo + eps)
    raise ValueError(f"unknown normalize_score method: {method!r}")


def smooth_score(score: mx.array, *, window: int = 1) -> mx.array:
    """C2's own required spec item: "specify smoothing." score:
    [batch, seq]. A causal moving average over the last `window`
    positions (including the current one) -- smooths single-position
    noise spikes without looking ahead (inference-safe, no future
    leakage). `window=1` (default) is the identity (no smoothing),
    preserving every existing caller's exact behavior."""
    if window <= 1:
        return score
    batch, seq = score.shape
    padded = mx.concatenate([mx.zeros((batch, window - 1)), score], axis=1)
    windows = mx.stack([padded[:, i:i + seq] for i in range(window)], axis=-1)
    counts = mx.array([min(i + 1, window) for i in range(seq)], dtype=mx.float32)[None, :]
    return mx.sum(windows, axis=-1) / counts


def rate_bounded_threshold(score: mx.array, *, target_rate: float, min_rate: float, max_rate: float) -> mx.array:
    """C2's own required spec item: "specify... thresholding, minimum
    and maximum trigger rates." score: [batch, seq] (typically already
    normalized). Returns a per-BATCH-ROW threshold (`[batch, 1]`) set
    via the `(1 - target_rate)`-quantile of that sequence's own score
    distribution, so triggering `score > threshold` yields close to
    `target_rate` fraction of positions triggered BY CONSTRUCTION,
    regardless of the score distribution's absolute scale -- clamped
    so the resulting rate cannot fall outside `[min_rate, max_rate]`
    (approximated by clamping `target_rate` itself into that range
    before computing the quantile, since the quantile-threshold
    construction makes the ACHIEVED rate track the TARGET rate
    directly). Fully deterministic -- no sampling, same output every
    call for the same input, satisfying C2's "deterministic inference
    behavior" requirement."""
    clamped_rate = min(max(target_rate, min_rate), max_rate)
    quantile = 1.0 - clamped_rate
    sorted_scores = mx.sort(score, axis=-1)
    seq = score.shape[-1]
    idx = min(max(int(quantile * seq), 0), seq - 1)
    return sorted_scores[:, idx:idx + 1]


def trigger_decision(score: mx.array, *, scale: mx.array, bias: mx.array, ste: bool = False) -> mx.array:
    """score: [batch, seq]. `scale`/`bias` are learned scalars
    calibrating the raw delta-norm (whose typical magnitude depends on
    `dim` and training dynamics) into a meaningful trigger logit --
    the only new learned parameters this signal needs. Returns a
    continuous trigger in (0,1) (or exactly {0,1} under `ste`, via the
    same straight-through pattern HZ-0B's write_gate already uses:
    `reference/hz0b_b8_latent_write.py`)."""
    logit = score * scale + bias
    soft = mx.sigmoid(logit)
    if not ste:
        return soft
    hard = (soft > 0.5).astype(mx.float32)
    return soft + mx.stop_gradient(hard - soft)


def masked_anchor_attention(x: mx.array, trigger: mx.array, *, qkv_w: mx.array, qkv_b: mx.array, out_w: mx.array, out_b: mx.array, heads: int) -> mx.array:
    """x: [batch, seq, dim], trigger: [batch, seq] in [0,1] (soft or
    hard). `qkv_w`/`out_w` use the standard `nn.Linear` weight layout
    ([out_features, in_features] -- e.g. pass `linear.weight` directly,
    not `linear.weight.T`); transposed internally, matching every other
    manual-matmul caller in this project (`reference/hz0b_memory_simulator.py`
    etc.) so callers don't have to remember an unusual convention.

    Standard causal multi-head attention, but BOTH the query and
    key/value sets are restricted to triggered positions via additive
    masking (large negative bias on any non-triggered key, and the
    OUTPUT zeroed at any non-triggered query position) -- exactly the
    semantics of "only triggered positions get to look back, and only
    at other triggered positions," computed via a full O(seq^2) score
    matrix for correctness/testability (C8's bounded kernel must match
    this reference, not the other way around)."""
    batch, seq, dim = x.shape
    head_dim = dim // heads
    qkv = x @ qkv_w.T + qkv_b
    q, k, v = mx.split(qkv.reshape(batch, seq, 3, heads, head_dim), 3, axis=2)
    q, k, v = (mx.squeeze(t, axis=2).transpose(0, 2, 1, 3) for t in (q, k, v))
    scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(head_dim, dtype=mx.float32))
    causal_mask = mx.triu(mx.full((seq, seq), -1e9), 1)
    key_trigger_mask = (1.0 - trigger)[:, None, None, :] * -1e9  # [batch,1,1,seq]
    scores = scores + causal_mask[None, None] + key_trigger_mask
    weights = mx.softmax(scores, axis=-1)
    out = mx.matmul(weights, v).transpose(0, 2, 1, 3).reshape(batch, seq, dim)
    out = out @ out_w.T + out_b
    return out * trigger[:, :, None]  # zero the contribution at non-triggered query positions


class SurpriseTriggeredBlock(nn.Module):
    """Drop-in replacement for `reference/hz0a_mlx_model.py::Block`
    when `anchor_capable=True` at a given layer index -- same
    `(dim, heads, d_ff, native_metal)` constructor shape and
    `__call__(x, state=None) -> (x, next_state)` interface, so it can
    be substituted per-layer exactly like the existing `attention`
    boolean already does, for C1's "three controlled models" (this
    IS model 3; model 1/2 are plain `Block` with
    `attention_indices=()` / the existing fixed schedule)."""

    def __init__(self, dim: int, heads: int, d_ff: int, native_metal: bool = False, ste: bool = False):
        super().__init__()
        self.ste = ste
        self.norm1, self.norm2 = nn.RMSNorm(dim), nn.RMSNorm(dim)
        self.recurrent = GDN2(dim, heads, native_metal)
        self.anchor_qkv = nn.Linear(dim, 3 * dim)
        self.anchor_out = nn.Linear(dim, dim)
        self.trigger_scale = mx.array([1.0])
        self.trigger_bias = mx.array([-1.0])
        self.gate, self.up, self.down = nn.Linear(dim, d_ff), nn.Linear(dim, d_ff), nn.Linear(d_ff, dim)
        self.heads = heads

    def __call__(self, x, state=None):
        normed1 = self.norm1(x)
        recurrent_out, next_state = self.recurrent(normed1, state)
        score = surprise_score(normed1)
        trigger = trigger_decision(score, scale=self.trigger_scale, bias=self.trigger_bias, ste=self.ste)
        anchor_out = masked_anchor_attention(
            normed1, trigger,
            qkv_w=self.anchor_qkv.weight, qkv_b=self.anchor_qkv.bias,
            out_w=self.anchor_out.weight, out_b=self.anchor_out.bias,
            heads=self.heads,
        )
        x = x + recurrent_out + anchor_out
        normed2 = self.norm2(x)
        mlp = self.down(nn.silu(self.gate(normed2)) * self.up(normed2))
        return x + mlp, next_state


class HZ0CSurpriseTriggeredModel(nn.Module):
    """C1's model 3: scaled recurrence with surprise-triggered anchors.
    Deliberately isolated from `reference/hz0a_mlx_model.py` rather
    than modifying `HZ0AMlxModel` to accept a pluggable block type --
    matches this project's established pattern (HZ-0B's B2 simulator
    was isolated until B6's integration phase) of not touching shared
    production code until an explicit integration phase says to.

    `anchor_indices`: which layers get `SurpriseTriggeredBlock`
    (recurrent GDN2 + conditional bounded anchor-attention) instead of
    a plain recurrent `Block` -- the direct C1 analog of
    `HZ0AMlxModel`'s `attention_indices`, at the SAME layer positions
    as model 2's fixed schedule for a layer-position-matched
    comparison. `attention_indices=()` for model 1 (no anchors at all)
    is just `HZ0AMlxModel` with an empty tuple -- no new class needed."""

    def __init__(self, vocab_size: int, dim: int, layers: int, heads: int, d_ff: int, anchor_indices: tuple[int, ...], native_metal: bool = False, ste: bool = False):
        super().__init__()
        self.vocab_size, self.dim, self.heads = vocab_size, dim, heads
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = [
            SurpriseTriggeredBlock(dim, heads, d_ff, native_metal, ste) if index in anchor_indices else Block(dim, heads, d_ff, False, native_metal)
            for index in range(layers)
        ]
        self.final_norm = nn.RMSNorm(dim)

    def __call__(self, token_ids, states=None):
        x = self.embedding(token_ids)
        if states is None:
            states = [None] * len(self.blocks)
        next_states = []
        for block, state in zip(self.blocks, states):
            x, state = block(x, state)
            next_states.append(state)
        return mx.matmul(self.final_norm(x), self.embedding.weight.T), next_states
