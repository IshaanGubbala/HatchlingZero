"""BDH-Delta ("Adaptive Evidence-Refresh BDH"), plans/newnewplan.md.

Real, direct implementation of the central architectural claim in that
plan: separate the EXPENSIVE exact addressing operation from CHEAP
internal computation, so recurrence stops being
"one reasoning step == one full exact re-query" (newnewplan.md section 2).

    e_j       = A(b_j, x)                                      exact, expensive, K times
    h_{j,0}   = I(b_j, e_j)
    h_{j,k+1} = h_{j,k} + alpha * g_{j,k} * Delta(h_{j,k}, e_j, b_j, q_{j,k})   cheap, M times per j
    b_{j+1}   = b_j + beta_scale * w_b * DeltaB(h_{j,M}, e_j, b_j)

A (evidence refresh) reuses the EXISTING exact-addressing pipeline
verbatim (encoder/attn/P/O/encoder_v -> decoder_up), unchanged from
reference/hz0h_bdh_vb_subspace_decoder_torch.py -- per newnewplan.md
section 14 ("keep this boring") and this project's own repeated,
real finding that every attempt to approximate/compress/route the
addressing side failed. decoder_up (nh*N -> subspace_rank) IS the
value-side evidence compression newnewplan.md section 10 asks for;
decoder_down (subspace_rank -> D) is reused as the evidence-expansion
operator inside I(b, e). Both stay SVD-warmstart-compatible with every
other variant in this project.

Addressing queries the BELIEF state b (not the fast scratch h) per
newnewplan.md's own e_j=A(b_j,x) -- belief is the more stable
representation, so what gets retrieved should track it, not the
volatile scratch state that resets every refresh block.

Real, disclosed simplifications made building this FIRST version
(newnewplan.md section 29's own words: "exact dimensions are
hypotheses, but the structure is what matters" -- these are the
places this file trades a literal reading for something buildable and
comparable to every other quality-check run in this project):

  - RMSNorm (as written in the plan) is replaced by this project's
    existing non-affine LayerNorm (`model.ln`), the one normalization
    op used everywhere else in this codebase. Both are just scale
    normalizations; using two different ones in the same model for no
    reason would be its own inconsistency.
  - Literal fixed "thinking register" slots (section 7, a separate
    S in R^{4x96}-style tensor) are NOT built as a distinct axis here.
    The D-wide scratch state h already IS a dense workspace the Think
    Cell reads/writes every microstep; carving out literal sub-slots
    within it is a real, separate follow-up, not attempted in v1.
  - Cross-token persistent latent carry (section 11: b_{t+1,0} =
    lambda*b_{t,final} + (1-lambda)*E(x_{t+1})) is NOT implemented.
    It needs a chunked/sequential execution mode; every other
    quality-check comparison in this project (Muon/MTP/n-gram/gated-
    residual/MoE/round-embed) trains on fixed-length sequences in one
    parallel forward, and this file keeps that convention so its
    val_loss stays comparable to all of them. A real gap, not a silent
    drop.
  - The predictor/corrector recurrence (section 25) and the evidence-
    disagreement compatibility score (section 26) are the plan's own
    "wild"/speculative additions, not part of its core claim -- both
    deferred.
  - "Evidence entropy" H(e_j) (section 5) is replaced with ||e_c||
    (L2 norm) as the state-of-computation feature -- e_c is a dense
    retrieved vector, not a probability distribution, so entropy isn't
    naturally defined for it.

Everything else in newnewplan.md's core claim IS implemented: decoupled
refresh cadence (n_refresh exact addresses, n_think cheap steps each),
the RMS-normalized gated delta update (section 4, directly targeting
the real R=12/16 collapse), two-timescale scratch/belief state (section
6), state-of-computation conditioning signals instead of round
identity (section 5, directly responding to the real round-embedding
negative result), no auxiliary state-target supervision (section 22,
directly responding to the real state-supervision shortcut result),
weight tying across every refresh/think step (section 24), packed
projections (section 4/16), dense-only static-unroll execution with
convergence-based update suppression rather than dynamic routing
(section 12, avoiding this project's own real gather/scatter-is-slower
finding), and protected near-zero-gate initialization (section 21,
same pattern validated by the Phase 4 single-gate result's g2=0.01
init).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder

_EPS = 1e-5


def add_delta_vnext(
    model: BDHVBSubspaceDecoder,
    n_refresh: int = 4,
    n_think: int = 2,
    think_hidden: int = 384,
    belief_hidden: int = 384,
    alpha_init: float = 0.5,
    beta_scale_init: float = 0.1,
    gamma_bias_init: float = -4.0,
    beta_bias_init: float = -5.0,
) -> None:
    C = model.config
    D = C.n_embd
    r = C.subspace_rank
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    model.n_refresh = n_refresh
    model.n_think = n_think
    model.think_hidden = think_hidden
    model.belief_hidden = belief_hidden

    # q (state-of-computation) features: ||delta_h_prev||, cos(h, h_prev),
    # ||e_c||, ||e_c - e_c_prev||, g_prev, c_prev -- 6 scalars, see module
    # docstring for why these replace round identity (newnewplan.md section 5).
    q_dim = 6
    think_in = D + r + D + q_dim  # h_norm, e_c, b, q
    think_out = 2 * think_hidden + 2  # u, v, gamma, c

    model.think_w_in = nn.Parameter(torch.zeros((think_in, think_out), device=device, dtype=dtype).normal_(std=0.02))
    model.think_b_in = nn.Parameter(torch.zeros((think_out,), device=device, dtype=dtype))
    with torch.no_grad():
        # Protected init (newnewplan.md section 21): gamma starts strongly
        # negative so g = sigmoid(gamma) ~ sigmoid(-4) ~ 0.018 at step 0 --
        # the Think Cell contributes almost nothing until it earns the
        # right to, same lesson as the validated g2=0.01 gated-residual init.
        model.think_b_in[2 * think_hidden] = gamma_bias_init
    model.think_w_out = nn.Parameter(torch.zeros((think_hidden, D), device=device, dtype=dtype).normal_(std=0.02))

    belief_in = D + r + D  # h_final_norm, e_c, b_norm
    belief_out = 2 * belief_hidden + 1  # u_b, v_b, beta
    model.belief_w_in = nn.Parameter(torch.zeros((belief_in, belief_out), device=device, dtype=dtype).normal_(std=0.02))
    model.belief_b_in = nn.Parameter(torch.zeros((belief_out,), device=device, dtype=dtype))
    with torch.no_grad():
        # Belief moves slower than scratch (beta << alpha) -- start it even
        # more suppressed than the scratch gate.
        model.belief_b_in[2 * belief_hidden] = beta_bias_init
    model.belief_w_out = nn.Parameter(torch.zeros((belief_hidden, D), device=device, dtype=dtype).normal_(std=0.02))

    model.think_alpha = nn.Parameter(torch.tensor(alpha_init, device=device, dtype=torch.float32))
    model.belief_beta_scale = nn.Parameter(torch.tensor(beta_scale_init, device=device, dtype=torch.float32))

    print(f"[delta_vnext] n_refresh={n_refresh} n_think={n_think} think_hidden={think_hidden} "
          f"belief_hidden={belief_hidden} alpha_init={alpha_init} beta_scale_init={beta_scale_init} "
          f"gamma_bias_init={gamma_bias_init} beta_bias_init={beta_bias_init}", flush=True)


def _rms(t: torch.Tensor) -> torch.Tensor:
    return t.pow(2).mean(dim=-1, keepdim=True).sqrt()


def _evidence_refresh(b: torch.Tensor, model: BDHVBSubspaceDecoder, B: int, T: int, D: int, nh: int, N: int) -> torch.Tensor:
    """e_j = A(b_j, x) -- exact BDH addressing, unchanged pipeline, querying
    belief instead of scratch. Identical FLOPs to one round of the base
    compound model (encoder -> attn -> O -> encoder_v -> decoder_up)."""
    x_sparse = F.relu(b @ model.encoder)
    v_bottleneck = b @ model.P
    yKV_bottleneck = model.attn(Q=x_sparse, K=x_sparse, V=v_bottleneck)
    yKV = model.ln(yKV_bottleneck @ model.O)
    y_latent = yKV @ model.encoder_v
    y_sparse = F.relu(y_latent)
    xy_sparse = model.drop(x_sparse * y_sparse)
    e_c = torch.matmul(xy_sparse, model.decoder_up.view(nh, N, -1)).sum(dim=1, keepdim=True)
    return e_c  # (B, 1, T, r)


def _think_step(h: torch.Tensor, e_c: torch.Tensor, b: torch.Tensor, q: torch.Tensor,
                 model: BDHVBSubspaceDecoder) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Dh = model.think_hidden
    inp = torch.cat([model.ln(h), e_c, b, q], dim=-1)
    packed = inp @ model.think_w_in + model.think_b_in
    u, v, gamma, c_logit = torch.split(packed, [Dh, Dh, 1, 1], dim=-1)
    delta_h = (F.silu(u) * v) @ model.think_w_out
    c = torch.sigmoid(c_logit)
    g = torch.sigmoid(gamma) * (1.0 - c)  # convergence suppresses its own update, section 12
    h_new = h + model.think_alpha * g * (delta_h / (_rms(delta_h) + _EPS))
    return h_new, g, c


def _belief_step(h_final: torch.Tensor, e_c: torch.Tensor, b: torch.Tensor,
                  model: BDHVBSubspaceDecoder) -> torch.Tensor:
    Dhb = model.belief_hidden
    inp = torch.cat([model.ln(h_final), e_c, model.ln(b)], dim=-1)
    packed = inp @ model.belief_w_in + model.belief_b_in
    u_b, v_b, beta_logit = torch.split(packed, [Dhb, Dhb, 1], dim=-1)
    delta_b = (F.silu(u_b) * v_b) @ model.belief_w_out
    w_b = torch.sigmoid(beta_logit)
    b_new = b + model.belief_beta_scale * w_b * (delta_b / (_rms(delta_b) + _EPS))
    return b_new


def _refresh_block(b: torch.Tensor, e_prev: torch.Tensor, delta_prev_norm: torch.Tensor,
                    cos_prev: torch.Tensor, g_prev: torch.Tensor, c_prev: torch.Tensor,
                    model: BDHVBSubspaceDecoder, B: int, T: int, D: int, nh: int, N: int):
    e_c = _evidence_refresh(b, model, B, T, D, nh, N)
    e_diff = torch.linalg.norm(e_c - e_prev, dim=-1, keepdim=True)
    e_norm = torch.linalg.norm(e_c, dim=-1, keepdim=True)

    h = model.ln(b + e_c @ model.decoder_down)  # I(b, e) -- fresh scratch each refresh block
    h_prev = h
    for _k in range(model.n_think):
        cos = F.cosine_similarity(h, h_prev, dim=-1).unsqueeze(-1)
        q = torch.cat([delta_prev_norm, cos, e_norm, e_diff, g_prev, c_prev], dim=-1)
        h_prev = h
        h_new, g, c = _think_step(h, e_c, b, q, model)
        delta_prev_norm = torch.linalg.norm(h_new - h, dim=-1, keepdim=True)
        h = h_new
        g_prev, c_prev = g, c

    b_new = _belief_step(h, e_c, b, model)
    return b_new, e_c, delta_prev_norm, cos, g_prev, c_prev


def bdh_delta_vnext_forward(
    model: BDHVBSubspaceDecoder,
    idx: torch.Tensor,
    n_refresh: int,
    targets: torch.Tensor | None = None,
):
    """n_refresh overrides model.n_refresh for the depth curriculum (same
    convention as every other checkpointed forward in this project, which
    takes a `depth`/`n_iterations` argument separate from config.n_layer).
    model.n_think is NOT curriculum-ramped -- only the number of expensive
    exact refreshes is, matching the plan's own framing of K (refresh
    count) as the expensive axis and M (think steps) as the cheap one."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    b = model.ln(model.embed(idx).unsqueeze(1))
    e_prev = torch.zeros(B, 1, T, C.subspace_rank, device=idx.device, dtype=b.dtype)
    delta_prev_norm = torch.zeros(B, 1, T, 1, device=idx.device, dtype=b.dtype)
    cos_prev = torch.ones(B, 1, T, 1, device=idx.device, dtype=b.dtype)
    g_prev = torch.zeros(B, 1, T, 1, device=idx.device, dtype=b.dtype)
    c_prev = torch.zeros(B, 1, T, 1, device=idx.device, dtype=b.dtype)

    for _j in range(n_refresh):
        b, e_prev, delta_prev_norm, cos_prev, g_prev, c_prev = torch.utils.checkpoint.checkpoint(
            _refresh_block, b, e_prev, delta_prev_norm, cos_prev, g_prev, c_prev,
            model, B, T, D, nh, N, use_reentrant=False,
        )

    logits = b.reshape(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return logits, loss
