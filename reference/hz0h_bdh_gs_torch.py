"""HZ-BDH-GS: Grouped Synaptic State (Phase 2R-C, `plans/HZ Phase 2R
State Redesign Plan.md`).

Real, explicit divergence from upstream BDH's STREAMING behavior only --
NOT a new model architecture and NOT a retrained variant. BDH's exact
streaming form (`reference/hz0h_bdh_torch.py`'s `bdh_stream_chunk`)
keeps one persistent state tensor PER LAYER, even though
`encoder`/`encoder_v`/`decoder` are already shared/tied across every
layer -- the state is the one thing that's still per-layer, which is
exactly why it's `n_layer` times bigger than it needs to be if layers
could share.

Key structural fact this file exploits: a state only matters ACROSS
STREAMING CALLS (it summarizes strictly-earlier time steps). A single
full-sequence forward pass (`BDH.forward`, or `bdh_stream_chunk` called
once with `start_position=0` on a freshly-initialized state) has NO
prior state to share -- the intra-chunk term alone covers the whole
sequence, and the cross-chunk term is exactly zero regardless of how
many state banks exist. That means grouping state banks changes
NOTHING about a single full-sequence forward pass, training included --
an already-trained exact-BDH model can be evaluated under grouped
streaming with ZERO retraining, and the exact-BDH oracle and every
group-count variant use the IDENTICAL weights. Only genuinely
multi-call streaming (chunked/token-by-token decode) is affected.

Design: `n_groups` state banks instead of `n_layer`, contiguous
depth-block assignment (layer `l` -> group `l * n_groups // n_layer`,
matching the user's own "depth 0,1,2 -> state A; depth 3,4,5 -> state B"
example). Every layer in a chunk-processing call reads its group's state
AS OF BEFORE THIS CHUNK (not updated mid-chunk by an earlier layer in
the same group -- a deliberate, disclosed design choice for
predictability); each layer's own chunk contribution (`K_t^T V_t`,
unaffected -- no per-layer read/write projections in this first version,
see the module-level TODO) is SUMMED across every layer in the group and
applied once, after all layers in the chunk have been processed. With
`n_groups == n_layer` (one group per layer, i.e. no sharing), this is
mathematically IDENTICAL to `bdh_stream_chunk` -- verified in
`tests/reference/test_hz0h_bdh_gs_torch.py`.

Real, deliberate scope limit vs. the user's own fuller sketch: no
per-layer read/write projections (`P_l`/`O_l`) yet -- this first version
tests whether PLAIN state-bank sharing (no extra learned disambiguation)
already preserves quality on its own. If it doesn't, that's the direct,
disclosed motivation for adding per-layer projections next (real future
work, not built here).
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH, Attention, BDHConfig


def layer_group_assignment(n_layer: int, n_groups: int) -> list[int]:
    """Contiguous depth blocks: layer l -> group l * n_groups // n_layer.
    E.g. n_layer=6, n_groups=2 -> [0,0,0,1,1,1] (depths 0-2 share bank 0,
    depths 3-5 share bank 1), matching the user's own example diagram."""
    if not (1 <= n_groups <= n_layer):
        raise ValueError(f"n_groups must be in [1, n_layer={n_layer}], got {n_groups}")
    return [layer * n_groups // n_layer for layer in range(n_layer)]


def init_bdh_grouped_states(model: BDH, n_groups: int, batch_size: int, device=None, dtype=None) -> list[torch.Tensor]:
    c = model.config
    nh, D = c.n_head, c.n_embd
    N = D * c.mlp_internal_dim_multiplier // nh
    device = device if device is not None else model.encoder.device
    dtype = dtype if dtype is not None else model.encoder.dtype
    return [torch.zeros(batch_size, nh, N, D, device=device, dtype=dtype) for _ in range(n_groups)]


def bdh_grouped_stream_chunk(
    model: BDH, group_states: list[torch.Tensor], idx_chunk: torch.Tensor, start_position: int, n_groups: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Same real per-layer computation as `bdh_stream_chunk` (identical
    intra-chunk term, identical use of shared encoder/encoder_v/decoder),
    except the CROSS-chunk term reads from a state bank shared by every
    layer in the same depth group (see `layer_group_assignment`), and
    each layer's own chunk contribution is summed into that shared bank
    (applied once, after every layer in this chunk has read the
    pre-chunk value -- see the module docstring for why)."""
    c = model.config
    B, L = idx_chunk.shape
    D = c.n_embd
    nh = c.n_head
    N = D * c.mlp_internal_dim_multiplier // nh
    device = idx_chunk.device
    layer_to_group = layer_group_assignment(c.n_layer, n_groups)

    x = model.embed(idx_chunk).unsqueeze(1)
    x = model.ln(x)

    positions = torch.arange(start_position, start_position + L, device=device, dtype=model.attn.freqs.dtype).view(1, 1, L, 1)
    r_phases = positions * model.attn.freqs

    group_contributions: dict[int, torch.Tensor] = {}
    for level in range(c.n_layer):
        group = layer_to_group[level]
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR
        V = x

        intra = (QR @ KR.mT).tril(diagonal=-1) @ V
        prefix_state = group_states[group]
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
        if group in group_contributions:
            group_contributions[group] = group_contributions[group] + chunk_contribution
        else:
            group_contributions[group] = chunk_contribution

    new_group_states = list(group_states)
    for group, contribution in group_contributions.items():
        new_group_states[group] = group_states[group] + contribution

    logits = x.view(B, L, D) @ model.lm_head
    return new_group_states, logits


# --- HZ-BDH-GSP: Grouped Synaptic State with per-layer Projections ---------
#
# The zero-shot functions above (`bdh_grouped_stream_chunk` etc.) tested
# PLAIN state-bank sharing on an already-trained exact-BDH model and found
# it fails badly (docs/restart/hz0h_phase2r_grouped_state_results.md):
# 2x reduction loses ~26% accuracy, 6x loses ~81%, monotonically worse
# with fewer groups. Summing different layers' `K^T V` contributions into
# one undifferentiated pool with no way to tell them apart destroys real
# information -- not a training-budget artifact (re-confirmed there via
# the same undertraining check used throughout Phase 2R).
#
# This section builds what the user's original 2R-C design actually
# specified from the start: small, PER-LAYER, TRAINED read/write
# projections (`P_l`/`O_l`, `D -> D`, one pair per layer, NOT tied like
# 2R-B's `P`/`O` were) so that layers sharing a state bank can still
# write/read it differently. `P_l` transforms the value BEFORE it enters
# the shared state (both the intra-chunk term and the persistent
# cross-chunk contribution); `O_l` transforms the combined attention
# output back out before it continues into the rest of that layer. A
# NEW trainable model class (`BDHGSP`) is required (unlike the zero-shot
# section above) because these projections must be learned jointly with
# the rest of the model, not applied post-hoc.
#
# Real, deliberate scope: no VALUE-WIDTH compression here (`P_l`/`O_l`
# are `D -> D`, not `D -> d_state`) -- this section is purely about
# fixing plain-grouping's information loss with per-layer
# disambiguation, so its own real effect can be measured in isolation
# before combining with 2R-B's value bottleneck in 2R-D, per the user's
# own "keeps the causal attribution clean" instruction.

@dataclasses.dataclass
class BDHGSPConfig(BDHConfig):
    n_state_groups: int = 0  # 0 -> defaults to n_layer (no sharing) in __post_init__

    def __post_init__(self) -> None:
        if self.n_state_groups == 0:
            self.n_state_groups = self.n_layer


class BDHGSP(torch.nn.Module):
    def __init__(self, config: BDHGSPConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        self.decoder = torch.nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = torch.nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = Attention(config)
        self.ln = torch.nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = torch.nn.Embedding(config.vocab_size, D)
        self.drop = torch.nn.Dropout(config.dropout)
        self.encoder_v = torch.nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.lm_head = torch.nn.Parameter(torch.zeros((D, config.vocab_size)).normal_(std=0.02))

        # The per-layer read/write projections -- the real addition vs
        # reference/hz0h_bdh_torch.py's BDH class. One independent P/O
        # pair per layer (NOT tied), unlike 2R-B's single shared P/O.
        self.P = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((D, D)).normal_(std=0.02)) for _ in range(config.n_layer)])
        self.O = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((D, D)).normal_(std=0.02)) for _ in range(config.n_layer)])
        self.group_of_layer = layer_group_assignment(config.n_layer, config.n_state_groups)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Deliberately NOT a single monolithic full-sequence computation
        (unlike BDH.forward or BDHVB.forward). A real design bug was
        caught before any training happened: an independent full-
        sequence attention pass per layer NEVER exercises cross-layer
        group sharing at all (a state only accumulates across separate
        streaming calls -- see the module docstring), so training that
        way would teach the model a computation it would never actually
        run at real (grouped-streaming) inference time -- a train/
        inference mismatch that would make any quality result from this
        class meaningless. Fixed by making `forward` itself run the
        IDENTICAL token-by-token mechanism `bdh_gsp_stream_chunk` uses at
        inference, so training genuinely sees and learns to use the
        shared group state. Real, disclosed cost: a Python loop over `T`
        positions instead of one batched matmul -- fine at this
        experiment's small scale, a real scaling concern for later,
        larger-scale Phase 2R work.
        """
        C = self.config
        B, T = idx.size()
        states = init_bdh_gsp_states(self, B, device=idx.device, dtype=self.encoder.dtype)
        all_logits = []
        for position in range(T):
            states, logits_t = bdh_gsp_stream_chunk(self, states, idx[:, position:position + 1], start_position=position)
            all_logits.append(logits_t)
        logits = torch.cat(all_logits, dim=1)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def init_bdh_gsp_states(model: BDHGSP, batch_size: int, device=None, dtype=None) -> list[torch.Tensor]:
    c = model.config
    nh, D = c.n_head, c.n_embd
    N = D * c.mlp_internal_dim_multiplier // nh
    device = device if device is not None else model.encoder.device
    dtype = dtype if dtype is not None else model.encoder.dtype
    return [torch.zeros(batch_size, nh, N, D, device=device, dtype=dtype) for _ in range(c.n_state_groups)]


def bdh_gsp_stream_chunk(
    model: BDHGSP, group_states: list[torch.Tensor], idx_chunk: torch.Tensor, start_position: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Same real streaming derivation as `bdh_grouped_stream_chunk`
    (shared state bank per depth group, every layer in a chunk reads the
    pre-chunk value, contributions summed and applied once after the
    chunk), except the value written to/read from the shared state goes
    through this layer's own trained `P_l`/`O_l` instead of the raw
    residual `x`."""
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

    group_contributions: dict[int, torch.Tensor] = {}
    for level in range(c.n_layer):
        group = model.group_of_layer[level]
        x_latent = x @ model.encoder
        x_sparse = F.relu(x_latent)
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR
        v = x @ model.P[level]

        intra = (QR @ KR.mT).tril(diagonal=-1) @ v
        prefix_state = group_states[group]
        cross = QR @ prefix_state
        yKV_raw = intra + cross
        yKV = yKV_raw @ model.O[level]
        yKV = model.ln(yKV)

        y_latent = yKV @ model.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, L, N * nh) @ model.decoder
        y = model.ln(yMLP)
        x = model.ln(x + y)

        chunk_contribution = KR.mT @ v
        if group in group_contributions:
            group_contributions[group] = group_contributions[group] + chunk_contribution
        else:
            group_contributions[group] = chunk_contribution

    new_group_states = list(group_states)
    for group, contribution in group_contributions.items():
        new_group_states[group] = group_states[group] + contribution

    logits = x.view(B, L, D) @ model.lm_head
    return new_group_states, logits
