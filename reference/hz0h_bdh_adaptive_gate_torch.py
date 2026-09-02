"""Adaptive Delta BDH, plans/newnewplan.md section 7-12 (the rewrite
after the cached-evidence crux, 2026-08-30). Section 28's experimental
order: (A) finish the refresh frontier [reference/hz0h_bdh_cached_evidence_torch.py],
(B) THIS FILE -- replace the single global g1 scalar with a tiny
state-dependent gate, tested in isolation at full refresh (8/8) against
the single-gate champion, before combining with any reduced-refresh
schedule (section 28D, not attempted here).

    h_{r+1} = LN(h_r + g_r * y_r),   g_r = g_theta(q_r)

y_r is EXACTLY the existing compound model's value/write computation
(encoder_v -> relu -> gate -> decoder_up compress -> decoder_down
expand) -- unchanged, per section 15's explicit warning not to replace
cached/existing computation with a new MLP. Only the residual-scale
gate becomes state-dependent instead of a single learned scalar.

Controller (section 8), deliberately tiny -- "tens or hundreds of
parameters", not another multi-million-parameter subsystem:

    q_r = [RMS(h_r), RMS(y_r), cos(h_r,y_r), RMS(h_r-h_{r-1}), d_r]
    g_r = sigmoid(W2 . SiLU(W1 . q_r + b1) + b2)

d_r (section 12) is the evidence-disagreement score, cos(h_r, e_r) --
computed directly with NO projection, unlike BDH-Delta's version:
e_r here is already full D-width (this architecture never compresses
evidence), so section 12's "avoid projections initially, if dimensions
align" applies cleanly.

Protected init (section 9): W2 initialized to EXACT zero (not just
small-std) and b2 = logit(0.58) -- the empirically-observed attractor
this session has now hit three independent times (0.583, 0.586,
0.5748). With W2=0, g_r = sigmoid(b2) = 0.58 EXACTLY, for every
position, completely independent of q_r, at step 0 -- training only
has to learn deviations from a working solution, never has to discover
the right operating point from scratch. W1/b1 get ordinary small-std
init so gradients can flow into them once W2 starts moving off zero.

Refresh cadence is handled exactly as reference/hz0h_bdh_cached_evidence_torch.py
(same refresh_schedule, same A_exact address / cached-iteration split,
reused not reimplemented) so this file works standalone for step B
(n_refresh=n_iterations, i.e. always refresh) and is already section
28D-ready (n_refresh<n_iterations) once step B has a verdict -- no
bundled experiment before an isolated win, per section 28's own rule.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_cached_evidence_torch import _address, refresh_schedule
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder

_EPS = 1e-5


def add_adaptive_gate(model: BDHVBSubspaceDecoder, hidden: int = 16, g_init: float = 0.58,
                       state_independent: bool = False) -> None:
    """state_independent=True builds the "killer control" the fixed-g1
    sweep's result demanded, 2026-08-31: the real result showed every
    frozen scalar (including g=0.55, the exact value the adaptive
    controller converged to) landed worse than the adaptive controller
    (1.4023) by a real +0.028 -- so the controller's OWN parameterization
    or training dynamics are doing something a fixed number can't, even
    though its measured output looks flat. This variant keeps the
    IDENTICAL controller (same width, same param count, same protected
    zero-init, same optimizer geometry) but feeds it a constant input
    (q := ones, ignoring h/y/e/h_prev entirely) instead of real state
    features -- g_r = C_theta(1), structurally incapable of varying by
    token/state/round no matter what it learns. Three possible outcomes:
    lands near 1.4023 -> the win is an optimization/parameterization
    effect, nothing to do with state-dependence; lands near 1.414 (the
    plain single-gate champion) -> state-dependence itself is what
    mattered; lands in between -> both contribute."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    q_dim = 5

    model.gate_hidden = hidden
    model.gate_state_independent = state_independent
    model.gate_w1 = nn.Parameter(torch.zeros((q_dim, hidden), device=device, dtype=dtype).normal_(std=0.02))
    model.gate_b1 = nn.Parameter(torch.zeros((hidden,), device=device, dtype=dtype))
    model.gate_w2 = nn.Parameter(torch.zeros((hidden, 1), device=device, dtype=dtype))  # exact zero: protected init
    logit = math.log(g_init / (1.0 - g_init))
    model.gate_b2 = nn.Parameter(torch.tensor(logit, device=device, dtype=torch.float32))

    n_gate_params = q_dim * hidden + hidden + hidden * 1 + 1
    print(f"[adaptive_gate] hidden={hidden} g_init={g_init} (logit={logit:.4f}) "
          f"controller_params={n_gate_params} state_independent={state_independent}", flush=True)


def _rms(t: torch.Tensor) -> torch.Tensor:
    """eps INSIDE the sqrt, not added after it: sqrt's gradient is
    infinite at exactly 0, and h-h_prev is EXACTLY zero at each refresh
    block's first iteration (h_prev is seeded from h itself before any
    update runs) -- a real NaN this project hit and fixed while building
    this file, not a hypothetical."""
    return (t.pow(2).mean(dim=-1, keepdim=True) + _EPS).sqrt()


def _adaptive_g(h: torch.Tensor, y: torch.Tensor, h_prev: torch.Tensor, e: torch.Tensor,
                 model: BDHVBSubspaceDecoder) -> torch.Tensor:
    if model.gate_state_independent:
        # C_theta(1): identical controller weights/optimizer geometry,
        # but the input is a constant -- structurally incapable of
        # varying by token/state/round, regardless of what it learns.
        q = torch.ones(*h.shape[:-1], 5, device=h.device, dtype=h.dtype)
    else:
        # e is (B, nh, T, D) here -- the attention output BEFORE the nh
        # dimension gets reduced (that reduction happens later, inside
        # _existing_compute_adaptive's decoder_up sum). d_r = cos(h, e) needs
        # a single D-wide evidence vector to compare against h (B,1,T,D), so
        # average across the nh "head" copies -- a real, disclosed necessity
        # (not in the plan's own notation, which doesn't have this per-head
        # structure), not a silent shortcut.
        e_summary = e.mean(dim=1, keepdim=True)
        q = torch.cat([
            _rms(h), _rms(y),
            F.cosine_similarity(h, y, dim=-1).unsqueeze(-1),
            _rms(h - h_prev),
            F.cosine_similarity(h, e_summary, dim=-1).unsqueeze(-1),
        ], dim=-1)
    hid = F.silu(q @ model.gate_w1 + model.gate_b1)
    g_logit = hid @ model.gate_w2 + model.gate_b2
    return torch.sigmoid(g_logit)


def _existing_compute_adaptive(x: torch.Tensor, x_sparse: torch.Tensor, e: torch.Tensor, h_prev: torch.Tensor,
                                model: BDHVBSubspaceDecoder, nh: int, N: int):
    y_latent = e @ model._w("encoder_v")
    y_sparse = F.relu(y_latent)
    xy_sparse = model.drop(x_sparse * y_sparse)
    alpha = torch.matmul(xy_sparse, model._w("decoder_up").view(nh, N, -1)).sum(dim=1, keepdim=True)
    y1 = model.ln(alpha @ model._w("decoder_down"))  # y_r
    g = _adaptive_g(x, y1, h_prev, e, model)
    x_new = model.ln(x + g * y1)
    return x_new, g


def _refresh_iteration(x: torch.Tensor, h_prev: torch.Tensor, model: BDHVBSubspaceDecoder, B: int, T: int, D: int, nh: int, N: int):
    e = _address(x, model, nh, N)
    x_new, g = _existing_compute_adaptive(x, F.relu(x @ model.encoder), e, h_prev, model, nh, N)
    return x_new, e, g


def _cached_iteration(x: torch.Tensor, e: torch.Tensor, h_prev: torch.Tensor, model: BDHVBSubspaceDecoder, B: int, T: int, D: int, nh: int, N: int):
    x_sparse = F.relu(x @ model.encoder)
    x_new, g = _existing_compute_adaptive(x, x_sparse, e, h_prev, model, nh, N)
    return x_new, e, g


def bdh_adaptive_gate_forward_checkpointed(
    model: BDHVBSubspaceDecoder,
    idx: torch.Tensor,
    n_iterations: int,
    n_refresh: int,
    targets: torch.Tensor | None = None,
):
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    refresh_at = refresh_schedule(n_iterations, n_refresh)

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)
    h_prev = x
    e = None
    for it in range(n_iterations):
        if it in refresh_at or e is None:
            x_new, e, g = torch.utils.checkpoint.checkpoint(_refresh_iteration, x, h_prev, model, B, T, D, nh, N, use_reentrant=False)
        else:
            x_new, e, g = torch.utils.checkpoint.checkpoint(_cached_iteration, x, e, h_prev, model, B, T, D, nh, N, use_reentrant=False)
        h_prev = x
        x = x_new

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return logits, loss
