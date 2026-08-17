"""Trainable exact-BDH executor with a persistent wide encoder parameter.

The oracle stores encoder as ``(H, D, N)``.  Its first projection broadcasts
one input across heads.  This executor instead stores the identical values in
CUDA/Tensor-Core-native ``(D, H*N)`` order, so each recurrent iteration uses
one large ``(B*T,D) @ (D,H*N)`` GEMM without rebuilding a permuted view.

This is opt-in and never modifies the pinned oracle.  Conversion helpers make
oracle checkpoints reversible, and tests cover forward, gradients, optimizer
updates, and round-trip state conversion.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_bmm_encoder_v_torch import bmm_encoder_v_step
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_wide_gemm_encoder_torch import bdh_wide_gemm_encoder_step


class WideParameterBDH(BDH):
    """Exact BDH whose encoder optimizer parameter is permanently ``(D,H*N)``.

    All values have a one-to-one correspondence with ``BDH.encoder``.  The
    shared recurrent use remains unchanged: this one parameter is read at
    every depth iteration.
    """

    def __init__(self, config: BDHConfig):
        super().__init__(config)
        h, d, n = self.encoder.shape
        wide = self.encoder.detach().permute(1, 0, 2).reshape(d, h * n).contiguous()
        del self.encoder
        self.encoder_wide = torch.nn.Parameter(wide)

    @property
    def encoder_oracle_view(self) -> torch.Tensor:
        """A non-copying logical ``(H,D,N)`` view for conversion/inspection."""
        c = self.config
        n = c.n_embd * c.mlp_internal_dim_multiplier // c.n_head
        return self.encoder_wide.reshape(c.n_embd, c.n_head, n).permute(1, 0, 2)

    @classmethod
    def from_oracle(cls, oracle: BDH) -> "WideParameterBDH":
        model = cls(oracle.config)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name == "encoder_wide":
                    parameter.copy_(oracle.encoder.permute(1, 0, 2).reshape_as(parameter))
                else:
                    parameter.copy_(dict(oracle.named_parameters())[name])
            model.attn.freqs.copy_(oracle.attn.freqs)
        return model

    def oracle_state_dict(self) -> dict[str, torch.Tensor]:
        """Export a normal oracle-compatible state dict without changing self."""
        state = {name: value.detach().clone() for name, value in self.state_dict().items()}
        state["encoder"] = self.encoder_oracle_view.detach().clone()
        del state["encoder_wide"]
        return state

    def load_oracle_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load a state dict emitted by the canonical ``BDH`` oracle."""
        expected = set(BDH(self.config).state_dict())
        if set(state) != expected:
            missing, extra = expected - set(state), set(state) - expected
            raise ValueError(f"oracle state keys mismatch; missing={missing}, extra={extra}")
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name == "encoder_wide":
                    parameter.copy_(state["encoder"].permute(1, 0, 2).reshape_as(parameter))
                else:
                    parameter.copy_(state[name])
            self.attn.freqs.copy_(state["attn.freqs"])

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        c = self.config
        b, t = idx.shape
        d, h = c.n_embd, c.n_head
        n = d * c.mlp_internal_dim_multiplier // h
        x = self.ln(self.embed(idx).unsqueeze(1))
        for _ in range(c.n_layer):
            # The parameter is already native wide layout: no per-forward
            # permute/contiguous conversion and no change to the dot products.
            x_latent = bdh_wide_gemm_encoder_step(x, self._w(self.encoder_wide), h, n)
            x_sparse = F.relu(x_latent)
            y_kv = self.ln(self.attn(Q=x_sparse, K=x_sparse, V=x))
            y_latent = bmm_encoder_v_step(y_kv, self._w(self.encoder_v))
            y_sparse = F.relu(y_latent)
            packed = self.drop(x_sparse * y_sparse).transpose(1, 2).reshape(b, 1, t, n * h)
            y = self.ln(packed @ self._w(self.decoder))
            x = self.ln(x + y)
        logits = x.view(b, t, d) @ self.lm_head
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
