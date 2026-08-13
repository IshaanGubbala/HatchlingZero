"""HZ Next-Phase Plan Phase B2 (`plans/HatchlingZero_Next_Phase_Plan.md`,
"Selective Synaptic State Writes"): a small, real, input-dependent gate
scaling how strongly each token's value gets written into the Value
Bottleneck's compressed synaptic state, gated on Phase B locking `D/4`
as the real quality/memory Pareto width (`docs/restart/hz0h_phase_b_vb_sweep_results.md`).

Real motivation: `reference/hz0h_bdh_vb_torch.py`'s VB compresses the
state's value width but still asks the model to write every token's
value through the same limited channel. The plan's real hypothesis: if
the model can learn WHICH tokens deserve to be written strongly into
that limited state, it may recover some of the quality VB gives up
under compression, without growing the state at all.

    g_t = sigmoid(W_g @ LN(x_t))                        # scalar per token
    S_t = S_{t-1} + K_t^T (g_t * P(x_t))                 # gated write

Real, load-bearing insight this file's implementation depends on: in
the STREAMING form, gating the write term `ΔS_t = K_t^T P(V_t)` by a
per-token scalar `g_t` is mathematically identical to gating the
bottlenecked value `P(V_t)` itself BEFORE the outer product (`K_t^T`
doesn't care whether the scalar is applied to its own factor or the
other one). That means this gate can be trained through
`BDHVB.forward`'s existing whole-sequence PARALLEL causal-attention
computation -- no token-by-token streaming loop needed for training,
unlike `reference/hz0h_bdh_gs_torch.py`'s `BDHGSP` (which genuinely
needed one, because ITS mechanism only manifests across separate
streaming calls). Verified directly here (see the test suite): the
parallel and streaming forms of this gated VB stay numerically
consistent, matching the same real property H2 established for exact
BDH's own state.
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import Attention, BDHConfig


@dataclasses.dataclass
class BDHVBSelectiveConfig(BDHConfig):
    d_state: int = 0  # 0 -> defaults to n_embd in __post_init__, same convention as BDHVBConfig

    def __post_init__(self) -> None:
        if self.d_state == 0:
            self.d_state = self.n_embd


class BDHVBSelective(torch.nn.Module):
    def __init__(self, config: BDHVBSelectiveConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        d_state = config.d_state

        self.decoder = torch.nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = torch.nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = Attention(config)
        self.ln = torch.nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = torch.nn.Embedding(config.vocab_size, D)
        self.drop = torch.nn.Dropout(config.dropout)
        self.encoder_v = torch.nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.lm_head = torch.nn.Parameter(torch.zeros((D, config.vocab_size)).normal_(std=0.02))

        # Value-bottleneck projections -- identical to BDHVB.
        self.P = torch.nn.Parameter(torch.zeros((D, d_state)).normal_(std=0.02))
        self.O = torch.nn.Parameter(torch.zeros((d_state, D)).normal_(std=0.02))

        # The one real addition vs BDHVB: a small, shared (tied across
        # every layer iteration, same convention as encoder/encoder_v/
        # decoder/P/O) gate mapping the current residual stream to a
        # scalar write-strength in (0, 1).
        self.write_gate = torch.nn.Parameter(torch.zeros((D, 1)).normal_(std=0.02))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        return bdh_vb_selective_forward(self, idx, targets=targets)

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


def compute_write_gate(model: "BDHVBSelective", x: torch.Tensor) -> torch.Tensor:
    """Real, differentiable per-token write-strength in (0, 1). `x` has
    shape (B, 1, T, D); returns (B, 1, T, 1), broadcastable against the
    d_state-wide bottlenecked value."""
    return torch.sigmoid(x @ model.write_gate)


def bdh_vb_selective_forward(model: BDHVBSelective, idx: torch.Tensor, targets: torch.Tensor | None = None, return_gates: bool = False):
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    all_gates = [] if return_gates else None

    for _level in range(C.n_layer):
        x_latent = x @ model.encoder
        x_sparse = F.relu(x_latent)

        v_bottleneck = x @ model.P  # (B, 1, T, d_state)
        gate = compute_write_gate(model, x)  # (B, 1, T, 1)
        if return_gates:
            all_gates.append(gate)
        v_bottleneck_gated = gate * v_bottleneck  # scales the WRITE, per token -- see module docstring for the streaming-form equivalence

        yKV_bottleneck = model.attn(Q=x_sparse, K=x_sparse, V=v_bottleneck_gated)
        yKV = yKV_bottleneck @ model.O
        yKV = model.ln(yKV)

        y_latent = yKV @ model.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)

        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder
        y = model.ln(yMLP)
        x = model.ln(x + y)

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    if return_gates:
        return logits, loss, all_gates
    return logits, loss


def init_bdh_vb_selective_states(model: BDHVBSelective, batch_size: int, device=None, dtype=None) -> list[torch.Tensor]:
    c = model.config
    nh, D = c.n_head, c.n_embd
    N = D * c.mlp_internal_dim_multiplier // nh
    device = device if device is not None else model.encoder.device
    dtype = dtype if dtype is not None else model.encoder.dtype
    return [torch.zeros(batch_size, nh, N, c.d_state, device=device, dtype=dtype) for _ in range(c.n_layer)]


def bdh_vb_selective_stream_chunk(
    model: BDHVBSelective, states: list[torch.Tensor], idx_chunk: torch.Tensor, start_position: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Real streaming form -- explicit per-layer state accumulator,
    written to token-by-token-equivalent within this chunk via the same
    real intra/cross-chunk split H2 established for exact BDH, with the
    write term gated exactly as `bdh_vb_selective_forward`'s parallel
    form gates it (mathematically the same operation, see module
    docstring). Used to VERIFY the parallel/streaming equivalence in
    tests, and as the real inference-time path once this mechanism is
    used for anything beyond training."""
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
        x_latent = x @ model.encoder
        x_sparse = F.relu(x_latent)
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR

        v_bottleneck = x @ model.P
        gate = compute_write_gate(model, x)
        v_bottleneck_gated = gate * v_bottleneck

        intra = (QR @ KR.mT).tril(diagonal=-1) @ v_bottleneck_gated
        prefix_state = states[level]
        cross = QR @ prefix_state
        yKV_bottleneck = intra + cross
        yKV = yKV_bottleneck @ model.O
        yKV = model.ln(yKV)

        y_latent = yKV @ model.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, L, N * nh) @ model.decoder
        y = model.ln(yMLP)
        x = model.ln(x + y)

        chunk_contribution = KR.mT @ v_bottleneck_gated
        new_states.append(states[level] + chunk_contribution)

    logits = x.view(B, L, D) @ model.lm_head
    return new_states, logits
