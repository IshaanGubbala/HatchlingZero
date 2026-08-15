"""HZ-CQ-0: the concrete first build of the S/H latent-reasoning
architecture (`plans/Deep Reserach Plan.md`'s "CQ-0: the concrete first
build, before any efficiency variant" section) -- a persistent
contextual state `S`, separate from an ephemeral, repeatedly-transformed
multi-slot reasoning workspace `H`, built directly on exact BDH's own
real streaming mechanics, not a new encoder invented from scratch.

Real motivation, restated from the plan doc: everything established
about Pathway's disclosed BDH-CQ interface -- separate persistent `S`
and ephemeral `H`, a structured multi-vector workspace, training
explicitly across multiple reasoning-effort levels -- is a REASONING
MECHANISM, not an efficiency property. Every efficiency variant built
this session (FoldBDH, Split-V, BlockBDH, activation checkpointing) is
correctly-labeled pre-CQ infrastructure work; none of it tests whether
this project can reproduce CQ's actual reasoning-scaling behavior. This
file is the first real attempt at that test.

Architecture, following the plan doc's own "Minimal HZ-CQ architecture"
pseudocode directly, not reinvented:

  S (persistent, demonstration-conditioned): exact BDH's own real
  per-layer streaming state, produced by `reference/hz0h_bdh_torch.py`'s
  `bdh_stream_chunk`/`init_bdh_states` -- NOT a new encoder. `S` is a
  list of `n_layer` tensors, one running state per BDH recurrent level,
  matching the plan doc's own "preserving the separate state bank
  associated with each recurrent BDH level" instruction (explicitly
  informed by the CLOSED grouped-recurrent-state experiment -- do not
  merge these banks).

  H (ephemeral, multi-slot workspace): `H in R^{B x M x D}`, `M` learned
  slot identities combined with a pooled query embedding at
  initialization. Per the plan doc's own pseudocode, one "reasoning
  cycle" walks through the SAME `n_layer` BDH levels (weight-shared,
  matching BDH's own tied-depth philosophy), reading the corresponding
  `S_l` at each level: `H = Phi_theta(H, Read(H, S_l))` for
  `l = 0..n_layer-1`. This whole walk repeats `R` times (the real
  reasoning-effort axis this file's own decisive gate is about).

  Read(H, S_l): H is projected into BDH's own query-latent space via a
  real learned encoder (`h_encoder`, same shape convention as BDH's own
  `encoder`: `(n_head, D, N)`), then reads S_l exactly the way BDH's own
  streaming cross-chunk term does (`Q @ S`, see
  `reference/hz0h_bdh_torch.py`'s `bdh_stream_chunk` for the oracle this
  mirrors) -- not an arbitrary new attention mechanism, the same
  operation BDH already uses to read its own state, applied to `H`
  instead of the token stream.

Real, disclosed simplifications for this MINIMAL first build (per the
plan doc's own "favor simplicity over cleverness" instruction) --
each a place a fuller HZ-CQ would likely differ:

  1. Read() combines per-head reads via a mean, not a learned mix --
     simplest correct choice, not claimed optimal.
  2. Output decoding reads directly from H's own M slots via a single
     shared linear decoder, not the plan's own eventual grid-cell-aware
     decoder/candidate-ranker (`GridDecoder`/`CandidateRanker` in the
     full pseudocode) -- ARC-specific decoding is explicitly deferred
     past CQ-0's own synthetic-task gate.
  3. `S` is never written to during the reasoning loop (matches the
     plan doc's own `dS/dr = 0` invariant under "Proposed S/H entity
     model") -- demonstrations are ingested once, before any reasoning
     cycle begins.

STATUS: architecture only, shape/gradient-correctness tested
(`tests/reference/test_hz0h_cq0_torch.py`). The actual CQ-0 gate itself
(`d(accuracy)/dR > 0`, growing with task dependency depth, on the
synthetic task ladder) requires a real task generator and training run,
neither built yet -- do not cite this file as evidence the CQ
mechanism works, only that it exists and computes without error.
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F
from torch import nn

from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_chunk, init_bdh_states


@dataclasses.dataclass
class HZCQ0Config:
    n_embd: int = 256
    n_layer: int = 4
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 16
    vocab_size: int = 256
    m_slots: int = 16
    dropout: float = 0.0


class HZCQ0(nn.Module):
    def __init__(self, config: HZCQ0Config):
        super().__init__()
        self.config = config
        D = config.n_embd
        nh = config.n_head
        N = D * config.mlp_internal_dim_multiplier // nh
        M = config.m_slots

        # S: exact BDH's own real streaming state -- not a new encoder.
        bdh_config = BDHConfig(
            n_layer=config.n_layer, n_embd=D, n_head=nh,
            mlp_internal_dim_multiplier=config.mlp_internal_dim_multiplier,
            vocab_size=config.vocab_size, dropout=config.dropout,
        )
        self.context_core = BDH(bdh_config)

        # H: learned initial slot identities, combined with a pooled
        # query embedding (reuses context_core's own embedding table --
        # one shared byte-level vocabulary for demos, query, and output).
        self.slot_embed = nn.Parameter(torch.zeros(M, D).normal_(std=0.02))
        self.query_proj = nn.Linear(D, D, bias=False)

        # Read(H, S_l): projects H into BDH's own query-latent space,
        # same (n_head, D, N) shape convention as BDH's own `encoder`.
        self.h_encoder = nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))

        # Phi_theta: shared update, reused every (r, l) step -- tied
        # depth, matching BDH's own shared-weight philosophy.
        self.update_proj = nn.Linear(2 * D, D, bias=False)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)

        # H_R -> output logits, per slot (minimal decoder, see module
        # docstring's disclosed simplification #2).
        self.decoder = nn.Parameter(torch.zeros(D, config.vocab_size).normal_(std=0.02))

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_state(self, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        return init_bdh_states(self.context_core, batch_size, device=device)

    def ingest(self, demo_tokens: torch.Tensor, states: list[torch.Tensor], start_position: int = 0) -> list[torch.Tensor]:
        """One real BDH streaming call over the demonstration tokens --
        `S` is exactly the per-layer running states this produces, not
        a separately-invented encoding."""
        states, _logits = bdh_stream_chunk(self.context_core, states, demo_tokens, start_position=start_position)
        return states

    def init_workspace(self, query_tokens: torch.Tensor, batch_size: int) -> torch.Tensor:
        q_embed = self.context_core.embed(query_tokens).mean(dim=1)  # (B, D)
        q_proj = self.query_proj(q_embed).unsqueeze(1)  # (B, 1, D)
        H = self.slot_embed.unsqueeze(0).expand(batch_size, -1, -1) + q_proj  # (B, M, D)
        return self.ln(H)

    def read(self, H: torch.Tensor, S_l: torch.Tensor) -> torch.Tensor:
        """`Q @ S`, the same real read operation
        `reference/hz0h_bdh_torch.py`'s `bdh_stream_chunk` already uses
        for its own cross-chunk term, applied to `H` instead of the
        token stream."""
        H4 = H.unsqueeze(1)  # (B, 1, M, D)
        H_latent = F.relu(H4 @ self.h_encoder)  # (B, nh, M, N)
        read = H_latent @ S_l  # (B, nh, M, N) @ (B, nh, N, D) -> (B, nh, M, D)
        return read.mean(dim=1)  # combine heads: simplest choice, see module docstring

    def reason_cycle(self, H: torch.Tensor, S: list[torch.Tensor]) -> torch.Tensor:
        """One full walk through the shared BDH levels -- the plan
        doc's own `H^{l+1} = Phi_theta(H^l, Read(H^l, S_l))` for
        `l = 0..n_layer-1`, all using the SAME `update_proj`/`h_encoder`
        weights (tied depth)."""
        for level in range(self.config.n_layer):
            read = self.read(H, S[level])
            update_in = torch.cat([H, read], dim=-1)  # (B, M, 2D)
            H = self.ln(H + self.update_proj(update_in))
        return H

    def forward(
        self,
        demo_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        r_iterations: int,
        targets: torch.Tensor | None = None,
    ):
        """`demo_tokens`: (B, T_demo), `query_tokens`: (B, T_query),
        `targets`: optional (B, M) -- one target token per output slot.
        `r_iterations`: number of full reasoning cycles (the real
        effort/depth axis; CQ-0's own gate is `d(accuracy)/d(r_iterations) > 0`,
        evaluated by calling this with different `r_iterations` on the
        SAME trained weights, not by training separate models)."""
        B = query_tokens.shape[0]
        device = query_tokens.device

        S = self.init_state(B, device)
        S = self.ingest(demo_tokens, S, start_position=0)
        # S is read-only from here on -- no write happens inside the
        # reasoning loop below, matching the plan doc's own dS/dr=0
        # invariant. Deliberately NOT detached: gradients should still
        # reach demonstration ingestion during training.

        H = self.init_workspace(query_tokens, B)
        for _r in range(r_iterations):
            H = self.reason_cycle(H, S)

        logits = H @ self.decoder  # (B, M, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
