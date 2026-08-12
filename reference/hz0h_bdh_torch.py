"""HZ-0H H1: faithful PyTorch port of the official BDH-GPU model.

REWRITTEN 2026-08-10/11 as a verbatim transcription of the real
`github.com/pathwaycom/bdh`'s `bdh.py` (fetched complete and verbatim,
not summarized, and diffed line-by-line against the prior hand-written
"port" here). The prior version had TWO real, confirmed bugs from
reading-and-retyping rather than copying exactly:

1. `Attention.phases_cos_sin` was missing the real `(phases % 1) * 2*pi`
   cycles->radians conversion -- confirmed to diverge from the real
   formula by up to ~2.0 (max possible for cos/sin) even at T=4. See
   `docs/restart/hz0h_rope_bug_critical_correction.md` for the full
   writeup and measured severity.
2. `BDH.__init__` never called `self.apply(self._init_weights)` (the
   real code's own override that sets `nn.Embedding`'s init to
   `std=0.02`, matching every other parameter's scale) -- `embed.weight`
   was left at PyTorch's default `nn.Embedding` init (`N(0,1)`, ~50x too
   large relative to every other parameter's `std=0.02`).

Below the `# --- REAL bdh.py, verbatim ---` marker is a byte-faithful
transcription (including comments, blank lines, and initialization
ORDER, since that affects RNG consumption for any given seed) of the
real source. This project's own real, necessary extensions (ternary
quantization support, H2's streaming/chunked functions) are added
BELOW that section, clearly separated, so the base stays a real,
directly-diffable copy of upstream rather than a hand-reconstruction
that can silently drift from it again.

This is an ISOLATED ORACLE: it does not touch, call, or depend on any
HZ-0A-G mechanism, and nothing in HZ's canonical backbone depends on
this file.

Two precise, real properties of the actual code (not bugs, deliberately
preserved, differ from the paper's simplified prose -- see
`docs/restart/hz0h_bdh_component_map.md`):

1. No softmax, no `K^T @ 1` normalization anywhere in attention -- raw
   `scores @ V`.
2. The causal mask is STRICTLY lower-triangular (`diagonal=-1`) -- a
   position cannot attend to itself, only to strictly earlier positions.

Weights are genuinely shared across depth: `encoder`, `encoder_v`,
`decoder`, and the single `ln` module are each ONE set of parameters
reused every iteration of the layer loop, not per-layer instances.
"""
from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn

# --- REAL bdh.py, verbatim (github.com/pathwaycom/bdh, fetched complete -----
# and diffed directly, not summarized). Only two changes from the literal
# upstream source: (1) BDHConfig gets one appended `ternary` field (see
# the HZ-0H T-lane extension below), (2) BDH.forward's three shared-weight
# reads (`self.encoder`/`self.encoder_v`/`self.decoder`) go through a
# `self._w(...)` hook to support ternary quantization -- both real,
# minimal, clearly-marked additions, not rewrites of the real logic.


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
    ternary: bool = False  # HZ-0H T-lane extension, not in the real upstream config; see docs/restart/hz0h_ternary_training_design.md


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q
        _, _, T, _ = Q.size()

        r_phases = (
            torch.arange(
                0,
                T,
                device=self.freqs.device,
                dtype=self.freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR

        # Current attention
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        C = self.config

        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)

        # actually helps with training
        x = self.ln(x)  # B, 1, T, D

        for level in range(C.n_layer):
            x_latent = x @ self._w(self.encoder)

            x_sparse = F.relu(x_latent)  # B, nh, T, N

            yKV = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
            )
            yKV = self.ln(yKV)

            y_latent = yKV @ self._w(self.encoder_v)
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse  # B, nh, T, N

            xy_sparse = self.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self._w(self.decoder)
            )  # B, 1, T, D
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- end of verbatim upstream source ---------------------------------------


# --- HZ-0H T-lane extension: ternary quantization (NOT in upstream) --------
# See docs/restart/hz0h_ternary_training_design.md for the T0 contract.
# `BDH.forward` above reads encoder/encoder_v/decoder through `self._w(...)`
# (the one real hook added into the verbatim forward loop above) so this
# can be toggled via `config.ternary` without touching upstream's own logic.

def quantize(t: torch.Tensor, q: int = 2) -> torch.Tensor:
    """Kept as a module-level function (distinct from get_freqs' own
    nested `quantize` closure above, which is verbatim upstream and left
    untouched) for reuse by anything importing this module expecting the
    old top-level name."""
    return (t / q).floor() * q


def _ternary_ste(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """BitNet b1.58-style absmean ternary quantization with a
    straight-through estimator -- same formula as
    `reference/hz0a_torch_model.py`'s `_ste_round_clip`, duplicated (not
    imported) rather than shared, to preserve this file's own "isolated
    oracle" contract. `w` stays full-precision as the actual
    `nn.Parameter`; only the VALUE USED in each forward call is
    re-quantized to `{-1, 0, 1} * gamma`, gradient is identity w.r.t. `w`
    via STE.
    """
    gamma = w.detach().abs().mean().clamp_min(eps)
    quantized = (w / gamma).round().clamp(-1, 1) * gamma
    return w + (quantized - w).detach()


def _bdh_w(self: BDH, param: torch.Tensor) -> torch.Tensor:
    """Returns `param` as-is, or its ternary-quantized value (STE) if
    `config.ternary` -- applied only to `encoder`/`encoder_v`/`decoder`
    per the T0 contract (`embed`/`lm_head`/`ln` stay full precision)."""
    return _ternary_ste(param) if self.config.ternary else param


BDH._w = _bdh_w


# --- H2: streaming/chunked state equivalence (NOT in upstream) -------------
#
# BDH-GPU's attention has no softmax and a STRICTLY causal mask (see the
# module docstring's discrepancy #1/#2): `scores = (QR @ KR^T).tril(-1);
# out = scores @ V`. Because there's no softmax normalization, this is
# secretly linear attention: for position t, `y_t = sum_{s<t} <QR_t, KR_s> V_s
# = QR_t @ S_t` where `S_t = sum_{s<t} KR_s (x) V_s` is a running
# outer-product state of shape (nh, N, D) per layer. That means BDH-GPU has
# an exact (not approximate) O(1)-state streaming/chunked form, the same
# "chunked linear attention" trick used for reference/hz0a_torch_model.py's
# GDN-2 mixer -- this is what plans/HZ-0H_BDH_Reconciliation_Plan.md's H2
# ("prove full-sequence, one-token, and arbitrary chunked streaming agree")
# asks to prove.
#
# One subtlety: RoPE's phase depends on the ABSOLUTE position of a token in
# the full sequence, not its position within a chunk -- `bdh_stream_chunk`
# takes an explicit `start_position` for this reason, so resuming a stream
# mid-sequence still uses the correct phases.
#
# Multi-layer note: each layer's `V = x` is that layer's OWN input (the
# residual stream value seen at this layer), and every layer collapses back
# to a single (B, 1, T, D) tensor via `decoder` before the next layer runs
# (the `nh` dim never survives past one layer's own attention block) -- so
# per-layer states are independent, each shaped (B, nh, N, D), and a
# strictly-in-order token/chunk traversal (finish all layers for token t
# before starting token t+1) has everything it needs at each layer: prior
# tokens have already been fully processed through every layer, including
# this one, by the time token t is processed.

def init_bdh_states(model: "BDH", batch_size: int, device=None, dtype=None) -> list[torch.Tensor]:
    """Fresh (all-zero) per-layer running states, one per `model.config.n_layer`,
    each shaped (batch_size, n_head, N, n_embd). Represents an empty prefix
    (no tokens streamed yet) -- pass this to start a new stream, or to reset
    one, since it captures no hidden information beyond its shape/dtype."""
    c = model.config
    nh, D = c.n_head, c.n_embd
    N = D * c.mlp_internal_dim_multiplier // nh
    device = device if device is not None else model.encoder.device
    dtype = dtype if dtype is not None else model.encoder.dtype
    return [torch.zeros(batch_size, nh, N, D, device=device, dtype=dtype) for _ in range(c.n_layer)]


def bdh_stream_chunk(
    model: "BDH", states: list[torch.Tensor], idx_chunk: torch.Tensor, start_position: int,
    diagnostics: dict | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Process one chunk of tokens (`idx_chunk`: (B, L), L >= 1) through the
    full model, given the running per-layer states from everything streamed
    so far and this chunk's absolute starting position. Returns
    `(new_states, logits)` with `logits` shaped (B, L, vocab_size);
    `new_states` folds in this chunk's own contribution and is ready to pass
    to the next call. L=1 is token-by-token streaming; L=T with a
    freshly-`init_bdh_states` state and `start_position=0` is mathematically
    identical to `BDH.forward`'s parallel computation (same formula, same
    operations -- the intra-chunk term alone covers the whole sequence and
    the cross-chunk term is exactly zero).

    `diagnostics`: optional dict, filled in place (not returned, so the
    return signature stays exactly what every existing caller/test already
    expects) with per-layer activation-sparsity (Phase 1 metric,
    plans/HatchlingZero_Reality_Plan.md) if passed. Reuses this function's
    own already-computed `x_sparse`/`y_sparse` rather than a separate
    reimplementation of the forward math -- `None` (the default) skips this
    entirely, zero cost/behavior change for existing callers.
    """
    if diagnostics is not None:
        diagnostics["x_sparse_fraction_zero"] = []
        diagnostics["y_sparse_fraction_zero"] = []
    c = model.config
    B, L = idx_chunk.shape
    D = c.n_embd
    nh = c.n_head
    N = D * c.mlp_internal_dim_multiplier // nh
    device = idx_chunk.device

    x = model.embed(idx_chunk).unsqueeze(1)
    x = model.ln(x)

    positions = torch.arange(start_position, start_position + L, device=device, dtype=model.attn.freqs.dtype).view(1, 1, L, 1)
    r_phases = positions * model.attn.freqs

    new_states = []
    for level in range(c.n_layer):
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)
        if diagnostics is not None:
            diagnostics["x_sparse_fraction_zero"].append(float((x_sparse == 0).float().mean()))
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR
        V = x

        intra = (QR @ KR.mT).tril(diagonal=-1) @ V
        prefix_state = states[level]
        cross = QR @ prefix_state
        yKV = intra + cross
        yKV = model.ln(yKV)

        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        if diagnostics is not None:
            diagnostics["y_sparse_fraction_zero"].append(float((y_sparse == 0).float().mean()))
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, L, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)

        chunk_contribution = KR.mT @ V
        new_states.append(prefix_state + chunk_contribution)

    logits = x.view(B, L, D) @ model.lm_head
    return new_states, logits


def new_bdh_kv_cache(model: "BDH") -> list[dict]:
    """One empty, mutable dict per layer -- pass to bdh_kv_cache_step,
    which grows each dict in place across repeated calls (same pattern
    as reference/hz0a_matched_transformer.py's MatchedTransformerLM.
    new_kv_cache)."""
    return [{} for _ in range(model.config.n_layer)]


def bdh_kv_cache_step(model: "BDH", kv_cache: list[dict], idx_token: torch.Tensor, position: int) -> torch.Tensor:
    """Alternative single-new-token decode path: an EXPLICIT, growing
    per-layer K/V cache (same style as a standard Transformer KV-cache),
    instead of `bdh_stream_chunk`'s O(1)-in-context, O(D^2)-per-layer
    compressed running state. Mathematically identical output (same real
    strictly-lower-triangular causal sum BDH.forward/bdh_stream_chunk
    compute, just accumulated a different way) -- see
    tests/reference/test_hz0h_bdh_kv_cache.py for the exact-agreement
    check against both.

    Real motivation (docs/restart/hz0h_phase1_crossover_scale_sweep_results.md):
    `bdh_stream_chunk`'s O(D^2)-per-layer state-update cost grows
    quadratically with model width and dominates at larger D, even
    though it's context-length-INDEPENDENT. This path instead costs
    O(D * context_so_far) per token per layer -- context-DEPENDENT (like
    a Transformer's KV-cache) but with a much smaller constant, since
    BDH's attention has no softmax (fewer ops per element than a
    Transformer's). At large D and moderate context, this may beat both
    `bdh_stream_chunk` AND a real Transformer KV-cache; at very long
    context it should eventually lose to `bdh_stream_chunk` again (its
    cost keeps growing with context, `bdh_stream_chunk`'s doesn't) --
    real, not yet measured, see the module-level TODO in
    scripts/hz0h_inference_benchmark.py.

    `idx_token`: (B, 1), exactly one new token. `position`: this token's
    absolute position in the full sequence (RoPE needs it, same reason
    `bdh_stream_chunk` takes `start_position`). Returns logits (B, 1,
    vocab_size). `kv_cache` is mutated in place: each layer's dict grows
    `"k"` (shape (B, nh, t, N)) and `"v"` (shape (B, 1, t, D), broadcast
    across heads, matching how `Attention.forward`'s own `V` is shared
    across heads) by one position per call.
    """
    C = model.config
    B = idx_token.shape[0]
    D = C.n_embd
    nh = C.n_head

    x = model.embed(idx_token).unsqueeze(1)  # (B, 1, 1, D)
    x = model.ln(x)

    position_tensor = torch.full((1, 1, 1, 1), float(position), device=idx_token.device, dtype=model.attn.freqs.dtype)
    r_phases = position_tensor * model.attn.freqs

    for level in range(C.n_layer):
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)
        QR = model.attn.rope(r_phases, x_sparse)  # (B, nh, 1, N)

        layer_cache = kv_cache[level]
        if "k" in layer_cache:
            scores = QR @ layer_cache["k"].mT  # (B, nh, 1, t)
            yKV = scores @ layer_cache["v"]  # (B, nh, 1, D)
            layer_cache["k"] = torch.cat([layer_cache["k"], QR], dim=2)
            layer_cache["v"] = torch.cat([layer_cache["v"], x], dim=2)
        else:
            # Strictly-lower-triangular causal mask (see
            # reference/hz0h_bdh_torch.py's module docstring, discrepancy
            # #2): the FIRST token has nothing strictly before it, so its
            # attention output is exactly zero -- not "attend to self,"
            # genuinely nothing. Real behavior, not a special case to
            # work around.
            yKV = torch.zeros(B, nh, 1, D, device=x.device, dtype=x.dtype)
            layer_cache["k"] = QR
            layer_cache["v"] = x
        yKV = model.ln(yKV)

        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)
        N = D * C.mlp_internal_dim_multiplier // nh
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, 1, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)

    return x.view(B, 1, D) @ model.lm_head


# --- Phase 3 (plans/HatchlingZero_Reality_Plan.md): synaptic-state ---------
# quantization (NOT in upstream). Direct response to
# docs/restart/hz0h_phase2_streaming_state_size_results.md's real, measured
# finding: BDH's fp32 running state is already 2-3.3x the size of the
# model's own weights at the scales tested. INT8 (the plan's own
# recommended first rung: "BF16 -> FP8/INT8 -> lower precision only if
# justified") is a direct 4x reduction vs fp32 -- cheapest real experiment
# to try first, per the plan's own "likely first combination to test" note
# (this file only does the quantization half; block-sparsity is separate,
# undone work).

def quantize_state_int8(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor absmax INT8 quantization (not per-block -- the plan's
    §6.4 recommends per-block scales for a production version; per-tensor
    is the simplest real first experiment). Returns (int8_tensor,
    scale) where `state ~= int8_tensor.float() * scale`."""
    scale = state.detach().abs().amax().clamp_min(1e-8) / 127.0
    quantized = (state / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def dequantize_state_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return quantized.to(torch.float32) * scale


def init_bdh_states_int8(model: "BDH", batch_size: int, device=None) -> list[dict]:
    """INT8-quantized-state analog of init_bdh_states -- a fresh (all-zero)
    state quantizes to zero exactly (no error at t=0), one dict per layer
    holding `{"q": int8 tensor, "scale": scale tensor}`."""
    fp32_states = init_bdh_states(model, batch_size, device=device, dtype=torch.float32)
    return [{"q": quantize_state_int8(state)[0], "scale": torch.tensor(1.0, device=state.device)} for state in fp32_states]


def bdh_stream_chunk_int8_state(
    model: "BDH", states: list[dict], idx_chunk: torch.Tensor, start_position: int,
) -> tuple[list[dict], torch.Tensor]:
    """Same real computation as `bdh_stream_chunk` (dequantize the
    incoming state, compute the exact same intra-chunk + cross-chunk sum,
    requantize the result) -- NOT a separate reimplementation of the
    forward math, just wraps quantize/dequantize around the state
    round-trip. Real, compounding quantization error is expected (each
    call requantizes the PREVIOUS call's already-quantized-and-noisy
    state, not the exact fp32 running sum) -- see
    docs/restart/hz0h_phase3_state_quantization_results.md for the real,
    measured quality impact on a real task, not assumed negligible.
    `states`: list of per-layer `{"q": ..., "scale": ...}` dicts (see
    `init_bdh_states_int8`), mutated by returning a NEW list (not
    in-place, matching `bdh_stream_chunk`'s own not-in-place contract for
    its `states` list)."""
    c = model.config
    B, L = idx_chunk.shape
    D = c.n_embd
    nh = c.n_head
    N = D * c.mlp_internal_dim_multiplier // nh
    device = idx_chunk.device

    x = model.embed(idx_chunk).unsqueeze(1)
    x = model.ln(x)

    positions = torch.arange(start_position, start_position + L, device=device, dtype=model.attn.freqs.dtype).view(1, 1, L, 1)
    r_phases = positions * model.attn.freqs

    new_states = []
    for level in range(c.n_layer):
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR
        V = x

        intra = (QR @ KR.mT).tril(diagonal=-1) @ V
        prefix_state = dequantize_state_int8(states[level]["q"], states[level]["scale"])
        cross = QR @ prefix_state
        yKV = intra + cross
        yKV = model.ln(yKV)

        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, L, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)

        chunk_contribution = KR.mT @ V
        new_state_fp32 = prefix_state + chunk_contribution
        new_q, new_scale = quantize_state_int8(new_state_fp32)
        new_states.append({"q": new_q, "scale": new_scale})

    logits = x.view(B, L, D) @ model.lm_head
    return new_states, logits


def compute_activation_and_state_diagnostics(model: "BDH", idx: torch.Tensor) -> dict:
    """Phase 1 metrics (plans/HatchlingZero_Reality_Plan.md): per-layer
    activation sparsity and synaptic-state norms, read-only (no_grad,
    doesn't disturb the caller's train/eval mode or gradients). Runs
    `idx` as ONE full-sequence chunk through `bdh_stream_chunk`
    (`start_position=0`, fresh `init_bdh_states`) -- per that function's
    own docstring, mathematically identical to `BDH.forward`'s parallel
    computation, so this measures the same activations `forward` would
    produce without a second, separately-written reimplementation of the
    forward math."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        states = init_bdh_states(model, idx.shape[0], device=idx.device)
        diagnostics: dict = {}
        new_states, _logits = bdh_stream_chunk(model, states, idx, start_position=0, diagnostics=diagnostics)
        state_norms = [float(s.norm()) for s in new_states]
        x_frac = diagnostics["x_sparse_fraction_zero"]
        y_frac = diagnostics["y_sparse_fraction_zero"]
    if was_training:
        model.train()
    return {
        "x_sparse_fraction_zero": x_frac,
        "y_sparse_fraction_zero": y_frac,
        "mean_activation_sparsity": sum(x_frac + y_frac) / len(x_frac + y_frac),
        "state_norms": state_norms,
        "mean_state_norm": sum(state_norms) / len(state_norms),
    }


def bdh_stream_sequence(
    model: "BDH", idx: torch.Tensor, chunk_sizes: list[int], states: list[torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Drive `bdh_stream_chunk` over arbitrary chunk boundaries. `chunk_sizes`
    must sum to `idx.shape[1]`; each may be any positive length (1 for pure
    token-by-token, or irregular sizes to test arbitrary chunk boundaries).
    `states` defaults to a fresh stream (`init_bdh_states`) -- pass a
    previously-returned state to resume mid-sequence."""
    B, T = idx.shape
    if sum(chunk_sizes) != T:
        raise ValueError(f"chunk_sizes must sum to sequence length {T}, got {sum(chunk_sizes)}")
    if states is None:
        states = init_bdh_states(model, B, device=idx.device)
    outputs = []
    position = 0
    for size in chunk_sizes:
        chunk = idx[:, position:position + size]
        states, logits = bdh_stream_chunk(model, states, chunk, start_position=position)
        outputs.append(logits)
        position += size
    return states, torch.cat(outputs, dim=1)
