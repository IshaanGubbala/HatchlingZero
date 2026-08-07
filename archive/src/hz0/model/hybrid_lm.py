from __future__ import annotations

import torch
from torch import nn

from .attn_residual import AttentionResidual
from .backends import BackendUnavailableError, UpstreamGDN2Mixer, gdn2_is_available
from .blocks import AnchorAttentionBlock, FeedForward, GDN2ReferenceMixerBlock, RMSNorm, RecurrentMixerBlock
from .session_scratchpad import ScratchpadLogEntry, SessionScratchpad


class HybridLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        use_attention: bool,
        mixer_backend: str,
    ) -> None:
        super().__init__()
        self.mixer = build_mixer(mixer_backend, d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.attention = AnchorAttentionBlock(d_model, n_heads, dropout) if use_attention else None
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mixer(x)
        if self.attention is not None:
            x = self.attention(x)
        return self.ffn(x)


class HybridLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        mixer_backend: str,
        attention_every: int,
        max_seq_len: int,
        scratchpad_slots: int = 0,
        scratchpad_momentum: float = 0.9,
        residual_mode: str = "standard",
        attn_res_rank: int | None = None,
        attn_res_heads: int = 1,
    ) -> None:
        super().__init__()
        if residual_mode not in {"standard", "attn_res"}:
            raise ValueError(f"Unknown residual_mode: {residual_mode}")
        self.residual_mode = residual_mode
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [
                HybridLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    use_attention=((idx + 1) % attention_every == 0),
                    mixer_backend=mixer_backend,
                )
                for idx in range(n_layers)
            ]
        )
        # One AttentionResidual module PER LAYER (not shared) -- matches
        # the layer-indexed alpha_{l,i} in the depth-attention formula:
        # each depth l learns its own preference over prior depths i<l,
        # not a single global depth-preference reused everywhere.
        self.attn_res_layers = (
            nn.ModuleList([AttentionResidual(d_model, rank=attn_res_rank, n_heads=attn_res_heads) for _ in range(n_layers)])
            if residual_mode == "attn_res"
            else None
        )
        self.norm = RMSNorm(d_model)
        self.scratchpad = SessionScratchpad(scratchpad_slots, d_model, momentum=scratchpad_momentum) if scratchpad_slots > 0 else None
        self.scratchpad_query = nn.Linear(d_model, d_model, bias=False) if self.scratchpad is not None else None
        self.scratchpad_key = nn.Linear(d_model, d_model, bias=False) if self.scratchpad is not None else None
        self.scratchpad_value = nn.Linear(d_model, d_model, bias=False) if self.scratchpad is not None else None
        self.scratchpad_gate = nn.Linear(d_model, d_model) if self.scratchpad is not None else None
        # LayerNorm on the routing side (input to scratchpad_query/key). The
        # post-backbone hidden state drifts in mean/amplitude as a function of
        # the filler span that sits between the (key, value) binding and the
        # query position. Normalizing before the routing projections makes the
        # routing key/query invariants more rotation- and scale-stable, so the
        # write at t=1 (value position) and the read at t=64 (key position)
        # can hit the same slot via ``slot_addresses @ scratchpad_key(z)``
        # even though the post-backbone magnitudes differ. Value and gate
        # projections stay on the raw (un-normalised) hidden state because the
        # model needs context-rich information there.
        self.scratchpad_norm = nn.LayerNorm(d_model) if self.scratchpad is not None else None
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, oracle_slot_schedule: torch.Tensor | None = None) -> torch.Tensor:
        x, _ = self.forward_with_optional_logs(
            tokens,
            return_scratchpad_logs=False,
            oracle_slot_schedule=oracle_slot_schedule,
        )
        return self.lm_head(x)

    def forward_with_optional_logs(
        self,
        tokens: torch.Tensor,
        *,
        return_scratchpad_logs: bool = False,
        oracle_slot_schedule: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[ScratchpadLogEntry]]:
        batch, seq = tokens.shape
        positions = torch.arange(seq, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)[None, :, :]
        if self.residual_mode == "attn_res":
            assert self.attn_res_layers is not None
            history = [x]
            for layer, attn_res in zip(self.layers, self.attn_res_layers):
                stacked = torch.stack(history, dim=2)  # [batch, seq, depth, d_model]
                layer_input = attn_res(stacked)
                x = layer(layer_input)
                history.append(x)
            x = history[-1]
        else:
            for layer in self.layers:
                x = layer(x)
        x = self.norm(x)
        logs: list[ScratchpadLogEntry] = []
        if self.scratchpad is not None:
            x, logs = self._apply_scratchpad(
                x,
                return_logs=return_scratchpad_logs,
                oracle_slot_schedule=oracle_slot_schedule,
            )
        return x, logs

    def _apply_scratchpad(
        self,
        x: torch.Tensor,
        *,
        return_logs: bool,
        oracle_slot_schedule: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[ScratchpadLogEntry]]:
        assert self.scratchpad is not None
        assert self.scratchpad_query is not None
        assert self.scratchpad_key is not None
        assert self.scratchpad_value is not None
        assert self.scratchpad_gate is not None
        assert self.scratchpad_norm is not None

        state = self.scratchpad.reset(batch_size=x.size(0), device=x.device, dtype=x.dtype)
        outputs = []
        logs: list[ScratchpadLogEntry] = []
        for t in range(x.size(1)):
            token_x = x[:, t]
            # Routing side: feed a LayerNorm-ed copy of the post-backbone
            # hidden state. Value / gate side: feed the raw context-rich
            # hidden state.
            routing_input = self.scratchpad_norm(token_x)
            oracle_slot_t = (
                oracle_slot_schedule[:, t].to(torch.long)
                if oracle_slot_schedule is not None
                else None
            )
            readout, state, entry = self.scratchpad.step(
                self.scratchpad_query(routing_input),
                self.scratchpad_key(routing_input),
                self.scratchpad_value(token_x),
                state,
                log=return_logs,
                oracle_slot=oracle_slot_t,
            )
            gate = torch.sigmoid(self.scratchpad_gate(token_x))
            outputs.append(token_x + gate * readout)
            if entry is not None:
                logs.append(entry)
        return torch.stack(outputs, dim=1), logs


def build_mixer(mixer_backend: str, d_model: int, n_heads: int, dropout: float) -> nn.Module:
    backend = mixer_backend.lower()
    if backend == "fallback":
        return RecurrentMixerBlock(d_model, dropout)
    if backend in {"gdn2_ref", "gdn2-reference", "reference_gdn2"}:
        return GDN2ReferenceMixerBlock(d_model, dropout)
    if backend == "gdn2":
        return UpstreamGDN2Mixer(d_model=d_model, n_heads=n_heads)
    if backend == "auto":
        available, _ = gdn2_is_available()
        if available:
            return UpstreamGDN2Mixer(d_model=d_model, n_heads=n_heads)
        return RecurrentMixerBlock(d_model, dropout)
    raise BackendUnavailableError(f"Unknown mixer backend: {mixer_backend}")
