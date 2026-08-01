"""Config-driven PyTorch HZ-0A reference model with recurrent state carry."""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class HZ0AConfig:
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    d_k: int
    d_v: int
    d_ff: int
    attention_layer_indices: tuple[int, ...]
    # "gdn2" (default, the current locked-spec mixer) or "gdn3" (the
    # candidate delta-rule mixer investigated in
    # docs/restart/hz0a_gdn3_candidate_design.md -- real, positive, but
    # still single-seed/small-scale evidence per that doc's own verdict,
    # not yet a validated replacement for HZ-0A's frozen Stage 2 spec).
    # Does not affect attention layers either way.
    mixer: str = "gdn2"

    @classmethod
    def from_json(cls, path: str | Path) -> "HZ0AConfig":
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(spec["vocab_size"], spec["d_model"], spec["num_layers"], spec["num_heads"], spec["head_dim_qk"], spec["head_dim_v"], spec["d_ff"], tuple(spec["attention_layer_indices"]), spec.get("mixer", "gdn2"))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(d_model, d_ff), nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


def _gdn2_step(state, decay_t, erase_t, write_t, v_t, k_t, q_t):
    state = decay_t[:, :, None, :] * (1 - erase_t[:, :, None, :]) * state + write_t[:, :, :, None] * v_t[:, :, :, None] * k_t[:, :, None, :]
    return state, torch.einsum("bhvk,bhk->bhv", state, q_t)


def _gdn2_sequential(state, decay, erase, write, v, k, q):
    """The exact same per-timestep recurrence as `_gdn2_step`, just with the
    `for t in range(steps)` loop moved inside a single function instead of
    living in `GDN2Mixer.forward`. Mathematically and numerically identical
    to the eager loop either way (no approximation, unlike `_gdn2_chunk`) --
    this only exists so a caller can `torch.compile` the *whole* per-chunk
    loop as one graph. Compiling one `_gdn2_step` call at a time (the
    original `--compile-step` approach) still dispatches `steps` separate
    Python-level calls per chunk; compiling this instead fuses all of them
    into a single call, which was measured to be faster for small-to-moderate
    `steps` (chunk_length up to ~16-32) -- beyond that the compiled graph
    gets large enough that per-call latency regresses (measured at steps=64),
    the same growing-compile-cost trend that made compiling an entire
    128+-step chunk (or the whole model) impractically slow in the first
    place, just showing up in the runtime too once the graph is big enough.
    """
    outputs = []
    for t in range(decay.shape[1]):
        state, out_t = _gdn2_step(state, decay[:, t], erase[:, t], write[:, t], v[:, t], k[:, t], q[:, t])
        outputs.append(out_t)
    return state, torch.stack(outputs, dim=1)


def _gdn2_chunk(state0, decay, erase, write, v, k, q):
    """Closed-form chunked evaluation of the GDN-2 recurrence, mathematically
    equivalent to running `_gdn2_step` sequentially over the `steps` dimension
    (verified to agree with the sequential loop to ~1e-6 in float32 forward
    and gradients). The recurrence's decay gate depends only on the key
    channel, not the value channel, so it is a per-channel linear (affine)
    scan and admits the standard chunked/parallel-scan reformulation used by
    fast gated-linear-attention kernels (e.g. GLA/DeltaNet): a cumulative
    log-decay `A_t = prod_{s<=t} a_s` turns the O(steps) sequential update
    into O(1) big matmuls plus a causal (steps x steps) mask, at the cost of
    computing ratios A_t/A_s that are unnecessary in the sequential form.

    KNOWN UNSAFE, NOT JUST "APPROXIMATE": this was validated against the
    sequential loop with decay/erase logits drawn from a narrow band around
    this layer's own bias initialization (decay~0.99, erase~0.01), where it
    agreed to ~1e-6 in float32. Under a freshly-initialized model's actual
    random `in_proj` weights (i.e. exactly the condition a real training run
    starts from), decay can saturate near 0 for some channels/timesteps, and
    the split-exponent trick this reformulation relies on (`q*A_t` and
    `k*A_s^-1` computed as separate factors before being multiplied back
    together) then produces individually-overflowing intermediates even
    though their product is mathematically bounded. Clamping the cumulative
    log-decay to a floor (below) prevents the resulting NaN/Inf, but the
    clamp itself then makes the *output* systematically wrong wherever the
    true cumulative decay was stronger than the clamp -- measured at ~20%
    mean relative gradient error against the sequential loop with real
    (untrained) weights, in BOTH float32 and bfloat16. That is a structural
    limitation of this specific reformulation applied to this recurrence
    (not a rounding-noise issue fixable by more precision), and it has NOT
    been shown safe for real training. Do not enable `--chunked-scan` for a
    training run whose results need to be trusted; it is kept here only as a
    documented, opt-in throughput experiment.
    """
    out_dtype = state0.dtype
    state0, decay, erase, write, v, k, q = (t.float() for t in (state0, decay, erase, write, v, k, q))
    steps = decay.shape[1]
    a = decay * (1 - erase)
    log_a = torch.log(a.clamp_min(1e-20))
    # Clamped to a floor (not just the per-step log above) because it is the
    # *cumulative* sum that can run away over many steps even when every
    # per-step term is individually finite -- unclamped, exp(-logA) can
    # overflow to inf (then inf-inf/0*inf -> NaN) whenever decay saturates
    # low for a sustained stretch, which the untrained/random-init weights
    # this was caught with do hit in practice. Clamping only discards the
    # contribution of channels that have already decayed past ~1e-30 of
    # their original magnitude, i.e. below float precision anyway.
    logA = torch.cumsum(log_a, dim=1).clamp_min(-70.0)
    A, invA = torch.exp(logA), torch.exp(-logA)
    wv = write * v

    q_scaled = q * A
    term1 = torch.einsum("bthk,bhvk->bthv", q_scaled, state0)

    k_scaled = k * invA
    causal = torch.tril(torch.ones(steps, steps, device=state0.device, dtype=torch.bool))
    attn = torch.einsum("bthk,bshk->bhts", q_scaled, k_scaled).masked_fill(~causal, 0.0)
    term2 = torch.einsum("bhts,bshv->bthv", attn, wv)
    out = term1 + term2

    ratio_to_end = torch.exp(logA[:, -1:, :, :] - logA)
    state_T = A[:, -1, :, None, :] * state0 + torch.einsum("bshk,bshv->bhvk", ratio_to_end * k, wv)
    return state_T.to(out_dtype), out.to(out_dtype)


def _gdn2_via_fla_gla(state0, decay, erase, write, v, k, q):
    """GDN-2's recurrence computed via `flash_linear_attention`'s `chunk_gla`
    Triton kernel, instead of this project's own hand-derived chunked-scan
    (`_gdn2_chunk`, which was measured unsafe -- see its docstring).

    The reduction to GLA is exact, not approximate: GDN-2's update
    `state_t = decay_t*(1-erase_t)*state_{t-1} + write_t*v_t (x) k_t` has
    the same shape as GLA's `state_t = exp(g_t)*state_{t-1} + k_t (x) v_t`
    once `g_t = log(decay_t*(1-erase_t))` (folding the erase gate into the
    decay) and `v_t` is pre-scaled by `write_t`. This was verified against
    this file's own sequential `_gdn2_step` loop -- not just on synthetic
    decay values but under this layer's actual bias-initialized regime --
    to ~0.2-0.3% mean relative gradient error in float32 and ~0.7-1.1% in
    bfloat16, with NO NaN/Inf and no systematic bias (unlike `_gdn2_chunk`,
    whose failure under real weights was a >1000x larger, structural error,
    not ordinary numerical noise). `flash_linear_attention` is a widely used,
    independently maintained library (not an in-house numerical derivation),
    used as the actual training backend for e.g. Kimi K3's KDA layers, which
    are close relatives of this same gated-delta-net family -- its chunked
    kernels are expected to already handle the intra-chunk log-decay
    stability that `_gdn2_chunk`'s naive split-exponent approach did not.

    Unlike `_gdn2_chunk` and `_gdn2_sequential`/`_seq_fn`, this does not
    need `--chunk-length`/`--truncate-backward` bookkeeping at all -- it
    processes the full sequence length in one call with no unrolled Python
    loop and no chunk-boundary state-detach machinery, since the underlying
    kernel is already a proper chunked/parallel implementation.
    """
    from fla.ops.gla import chunk_gla
    out_dtype = state0.dtype
    g = torch.log((decay.float() * (1 - erase.float())).clamp_min(1e-20)).to(decay.dtype)
    v_scaled = write * v
    initial_state = state0.transpose(-1, -2).float().contiguous() if state0 is not None else None
    out, final_state = chunk_gla(q, k, v_scaled, g, scale=1.0, initial_state=initial_state, output_final_state=True)
    return final_state.transpose(-1, -2).to(out_dtype), out.to(out_dtype)


class GDN2Mixer(nn.Module):
    # `_seq_fn` is a class attribute (not per-instance) so a caller can swap in a
    # `torch.compile`-wrapped version once and have every layer instance share the
    # compiled artifact (same bytecode + shapes -> no per-layer recompilation).
    # Default is the plain eager loop; `--compile-step` swaps in a compiled
    # `_gdn2_sequential` (still exact, see its docstring for why compiling the
    # whole per-chunk loop beats compiling one step at a time).
    _seq_fn = staticmethod(_gdn2_sequential)
    # Same sharing rationale as `_seq_fn`, for the chunked-scan fast path.
    _chunk_fn = staticmethod(_gdn2_chunk)
    # Same sharing rationale, for the flash-linear-attention-backed fast path.
    _fla_fn = staticmethod(_gdn2_via_fla_gla)
    # Opt-in fast path (see `_gdn2_chunk` docstring for the numerical trade-off);
    # off by default so existing behavior/parity is untouched unless requested.
    _use_chunked_scan = False
    # Opt-in fast path (see `_gdn2_via_fla_gla` docstring) -- validated safe
    # under this layer's real init, unlike `_use_chunked_scan`, but still off
    # by default until proven out end-to-end (full-model parity + a training
    # trajectory comparison), consistent with how every other math-changing
    # path in this file was introduced.
    _use_fla = False

    def __init__(self, c: HZ0AConfig):
        super().__init__()
        self.c = c
        width = c.num_heads * (4 * c.d_k + 2 * c.d_v)
        self.in_proj, self.out_proj = nn.Linear(c.d_model, width), nn.Linear(c.num_heads * c.d_v, c.d_model)
        start = c.num_heads * (2 * c.d_k + c.d_v)
        self.in_proj.bias.data[start:start + c.num_heads * c.d_k].fill_(4.59512)
        self.in_proj.bias.data[start + c.num_heads * c.d_k:start + 2 * c.num_heads * c.d_k].fill_(-4.59512)
        self.in_proj.bias.data[start + 2 * c.num_heads * c.d_k:].fill_(-4.59512)

    def forward(self, x, state):
        c, bsz, steps = self.c, x.shape[0], x.shape[1]
        p = self.in_proj(x).view(bsz, steps, c.num_heads, 4 * c.d_k + 2 * c.d_v)
        q, k, v = p[..., :c.d_k], p[..., c.d_k:2*c.d_k], p[..., 2*c.d_k:2*c.d_k+c.d_v]
        offset = 2 * c.d_k + c.d_v
        decay, erase, write = torch.sigmoid(p[..., offset:offset+c.d_k]), torch.sigmoid(p[..., offset+c.d_k:offset+2*c.d_k]), torch.sigmoid(p[..., offset+2*c.d_k:])
        if type(self)._use_fla:
            state, out = type(self)._fla_fn(state, decay, erase, write, v, k, q)
        elif type(self)._use_chunked_scan:
            state, out = type(self)._chunk_fn(state, decay, erase, write, v, k, q)
        else:
            state, out = type(self)._seq_fn(state, decay, erase, write, v, k, q)
        return self.out_proj(out.reshape(bsz, steps, c.num_heads * c.d_v)), state


class CausalAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.qkv, self.out = nn.Linear(c.d_model, 3 * c.num_heads * c.d_k), nn.Linear(c.num_heads * c.d_k, c.d_model)

    def forward(self, x):
        c, bsz, steps = self.c, x.shape[0], x.shape[1]
        q, k, v = self.qkv(x).view(bsz, steps, c.num_heads, 3 * c.d_k).chunk(3, dim=-1)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1, 2).reshape(bsz, steps, c.num_heads * c.d_k))


def _build_recurrent_mixer(c):
    if c.mixer == "gdn3":
        from reference.hz0a_gdn3_candidate_mixer_torch import GDN3CandidateMixerTorch
        assert c.d_k == c.d_v, "GDN3CandidateMixerTorch assumes a single head_dim (d_k == d_v)"
        return GDN3CandidateMixerTorch(c.d_model, c.num_heads, head_dim=c.d_k)
    if c.mixer != "gdn2":
        raise ValueError(f"unknown mixer {c.mixer!r}, expected 'gdn2' or 'gdn3'")
    return GDN2Mixer(c)


class HZ0ABlock(nn.Module):
    def __init__(self, c, attention):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.mixer = CausalAttention(c) if attention else _build_recurrent_mixer(c)
        self.mlp, self.attention = SwiGLU(c.d_model, c.d_ff), attention

    # Opt-in: recompute the MLP (norm2+SwiGLU) during backward instead of
    # keeping its activations resident. Unlike `torch.compile`, this changes
    # WHEN the forward math runs, not WHAT it computes, so it is bit-exact
    # with the non-checkpointed path (verified: 0.0 diff) -- the only
    # tradeoff is compute (one extra forward pass through the MLP) for
    # memory. Off by default; see `--activation-checkpoint` in the runner
    # for the measured tradeoff on this hardware (net positive here, unlike
    # the Mac/MLX runner's own `--activation-checkpoint`, which regressed
    # throughput 16% at this model scale -- a different framework/kernel,
    # not assumed to transfer, and independently re-measured for this path).
    _checkpoint_mlp = False

    def forward(self, x, state):
        mixed = self.mixer(self.norm1(x)) if self.attention else self.mixer(self.norm1(x), state)
        if self.attention:
            mixed, next_state = mixed, None
        else:
            mixed, next_state = mixed
        x = x + mixed
        if type(self)._checkpoint_mlp:
            return x + checkpoint(lambda z: self.mlp(self.norm2(z)), x, use_reentrant=False), next_state
        return x + self.mlp(self.norm2(x)), next_state


class HZ0AModel(nn.Module):
    def __init__(self, c: HZ0AConfig):
        super().__init__()
        self.config, self.embedding = c, nn.Embedding(c.vocab_size, c.d_model)
        # PyTorch's nn.Embedding default init is N(0, 1) (std=1.0) --
        # ~28x larger than MLX's own default (`scale = sqrt(1/dims)`,
        # reference/hz0a_mlx_model.py via mlx.nn.Embedding), which this
        # weight-tied embedding/LM-head table is meant to match. Left at
        # the torch default, this silently made every downstream
        # activation and gradient ~20-40x too large from step 1 --
        # found 2026-08-01 while diagnosing a persistently 5-7x elevated
        # loss on the RTX 3060 Stage 2 replication run (real gradient
        # norms observed: 300-1000+ vs. the native MLX runner's own
        # logged 3-25 over the same early-training window). Not a
        # cosmetic fix -- this is the actual root cause, not a tuning
        # knob.
        nn.init.normal_(self.embedding.weight, std=math.sqrt(1.0 / c.d_model))
        self.blocks = nn.ModuleList(HZ0ABlock(c, i in c.attention_layer_indices) for i in range(c.num_layers))
        self.final_norm = RMSNorm(c.d_model)

    def init_states(self, batch_size, device=None, dtype=None):
        c = self.config
        return [None if block.attention else torch.zeros(batch_size, c.num_heads, c.d_v, c.d_k, device=device, dtype=dtype or self.embedding.weight.dtype) for block in self.blocks]

    def forward(self, token_ids, states=None):
        x = self.embedding(token_ids)
        states = self.init_states(token_ids.shape[0], token_ids.device) if states is None else states
        next_states = []
        for block, state in zip(self.blocks, states):
            x, state = block(x, state)
            next_states.append(state)
        return torch.einsum("btd,vd->btv", self.final_norm(x), self.embedding.weight), next_states


def parameter_count(config: HZ0AConfig) -> int:
    return sum(parameter.numel() for parameter in HZ0AModel(config).parameters())
