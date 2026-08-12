"""HZ Phase 2R Step 2 (`plans/HZ Integrated Candidate Plan.md`): the one
authorized redesign attempt for grouped state, after `reference/hz0h_bdh_gs_torch.py`'s
`BDHGSP` (hard depth-block group assignment + per-layer D->D
projections) hit a genuine optimization plateau -- 4 mechanistically
distinct training procedures (full BPTT, 4x LR, sequence curriculum,
truncated BPTT) all converged to the SAME ~44% ceiling on passkey, even
at n_state_groups=6 (no sharing at all), which means the plateau is at
least partly a limitation of the per-layer-projection FORMULATION
itself, not (only) the grouping.

Per the plan's own Step 2 spec verbatim: "give each layer a small
learned addressing vector a_l over k shared memory banks S_1...S_k,
with soft read/write routing S_l^read = sum_j p_lj * S_j (p_l =
softmax(a_l) or similar). Depths share the expensive memory but keep a
learned identity within it, instead of being forced into one
undifferentiated pool." Deliberately does NOT add BDHGSP's P_l/O_l
value-space projections -- this tests the plan's OWN new mechanism
(learned soft addressing) in isolation, not addressing-plus-projections,
so a plateau here can be attributed to the addressing idea itself.

Per the plan: "If this also plateaus at the same loss floor pattern
2R-C showed: kill grouped-state compression entirely... Do not keep
iterating past one real redesign attempt."
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import Attention, BDHConfig


@dataclasses.dataclass
class BDHSoftGroupedConfig(BDHConfig):
    n_state_banks: int = 0  # 0 -> defaults to n_layer (learned addressing over n_layer banks, not hard no-sharing)

    def __post_init__(self) -> None:
        if self.n_state_banks == 0:
            self.n_state_banks = self.n_layer


class BDHSoftGroupedState(torch.nn.Module):
    def __init__(self, config: BDHSoftGroupedConfig):
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

        # The real addition vs BDH: a learned addressing vector a_l per
        # layer over n_state_banks shared banks. p_l = softmax(a_l) is
        # recomputed from these logits every forward call (not fixed),
        # so gradient can genuinely reshape each layer's learned identity
        # within the shared pool.
        self.address_logits = torch.nn.Parameter(torch.zeros((config.n_layer, config.n_state_banks)).normal_(std=0.02))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Token-by-token, exactly like BDHGSP.forward -- a state only
        accumulates across separate streaming calls (see
        reference/hz0h_bdh_gs_torch.py's module docstring, and the real
        design bug it caught before any training happened); a single
        batched full-sequence pass would never exercise the shared banks
        at all."""
        C = self.config
        B, T = idx.size()
        states = init_bdh_soft_grouped_states(self, B, device=idx.device, dtype=self.encoder.dtype)
        all_logits = []
        for position in range(T):
            states, logits_t = bdh_soft_grouped_stream_chunk(self, states, idx[:, position:position + 1], start_position=position)
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


def init_bdh_soft_grouped_states(model: BDHSoftGroupedState, batch_size: int, device=None, dtype=None) -> list[torch.Tensor]:
    c = model.config
    nh, D = c.n_head, c.n_embd
    N = D * c.mlp_internal_dim_multiplier // nh
    device = device if device is not None else model.encoder.device
    dtype = dtype if dtype is not None else model.encoder.dtype
    return [torch.zeros(batch_size, nh, N, D, device=device, dtype=dtype) for _ in range(c.n_state_banks)]


def bdh_soft_grouped_stream_chunk(
    model: BDHSoftGroupedState, bank_states: list[torch.Tensor], idx_chunk: torch.Tensor, start_position: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Same real per-layer computation as `bdh_grouped_stream_chunk`
    (identical intra-chunk term, identical shared encoder/encoder_v/
    decoder), except the cross-chunk term each layer reads is a SOFT,
    learned combination of ALL k banks (`sum_j p_lj * S_j`, `p_l =
    softmax(address_logits[l])`) rather than one hard-assigned bank, and
    each layer's own chunk contribution is written into EVERY bank,
    weighted by that same `p_l` -- a real, differentiable read/write
    routing, not a fixed partition."""
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

    p = F.softmax(model.address_logits, dim=-1)  # (n_layer, n_state_banks)
    bank_contributions: list[torch.Tensor | None] = [None] * c.n_state_banks

    for level in range(c.n_layer):
        x_latent = x @ model.encoder
        x_sparse = F.relu(x_latent)
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR
        V = x

        intra = (QR @ KR.mT).tril(diagonal=-1) @ V
        prefix_state = sum(p[level, j] * bank_states[j] for j in range(c.n_state_banks))
        cross = QR @ prefix_state
        yKV = intra + cross
        yKV = model.ln(yKV)

        y_latent = yKV @ model.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, L, N * nh) @ model.decoder
        y = model.ln(yMLP)
        x = model.ln(x + y)

        chunk_contribution = KR.mT @ V
        for j in range(c.n_state_banks):
            weighted = p[level, j] * chunk_contribution
            bank_contributions[j] = weighted if bank_contributions[j] is None else bank_contributions[j] + weighted

    new_bank_states = [bank_states[j] + bank_contributions[j] for j in range(c.n_state_banks)]

    logits = x.view(B, L, D) @ model.lm_head
    return new_bank_states, logits
