"""Training-only value-path LoRA for the current HZ-CQ core.

Targets ``encoder_v``, ``decoder_up``, and ``decoder_down`` only.  The
addressing projection and the tiny adaptive controller are deliberately not
adapted.  All B factors start at zero, therefore both regular forward and
``forward_hz_cq`` are exactly unchanged at initialization or scale zero.
"""
from __future__ import annotations
import math
from collections.abc import Iterable
import torch
from torch import nn
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig

_ALLOWED = frozenset(("encoder_v", "decoder_up", "decoder_down"))

class HZCQReasoningLoRA(BDHVBSubspaceDecoder):
    def __init__(self, config: BDHVBSubspaceDecoderConfig, *, rank: int = 8,
                 alpha: float | None = None,
                 targets: Iterable[str] = ("encoder_v", "decoder_up", "decoder_down"),
                 freeze_base: bool = False):
        if rank < 1: raise ValueError("rank must be positive")
        targets = tuple(targets)
        if not targets or set(targets) - _ALLOWED:
            raise ValueError(f"targets must be a nonempty subset of {sorted(_ALLOWED)}")
        super().__init__(config)
        self.rank, self.lora_alpha, self.targets = rank, float(rank if alpha is None else alpha), frozenset(targets)
        self.register_buffer("lora_scale", torch.tensor(1.0), persistent=True)
        self._merged = False
        nh, D = config.n_head, config.n_embd
        N, r_dec = D * config.mlp_internal_dim_multiplier // nh, config.subspace_rank
        shapes = {"encoder_v": (nh, D, N), "decoder_up": (nh, N, r_dec), "decoder_down": (1, r_dec, D)}
        for name in self.targets:
            heads, inp, out = shapes[name]
            setattr(self, f"lora_{name}_A", nn.Parameter(torch.empty(heads, inp, rank)))
            setattr(self, f"lora_{name}_B", nn.Parameter(torch.zeros(heads, rank, out)))
            nn.init.kaiming_uniform_(getattr(self, f"lora_{name}_A"), a=math.sqrt(5))
        if freeze_base: self.freeze_base_parameters()

    def _delta(self, name: str) -> torch.Tensor:
        return (getattr(self, f"lora_{name}_A") @ getattr(self, f"lora_{name}_B")) * (self.lora_scale * self.lora_alpha / self.rank)

    def _w(self, name: str) -> torch.Tensor:
        base = getattr(self, name)
        if self._merged or name not in self.targets or float(self.lora_scale.detach()) == 0.0: return base
        delta = self._delta(name).to(base.dtype)
        return base + (delta.reshape_as(base) if name != "decoder_down" else delta.squeeze(0))

    @torch.no_grad()
    def set_lora_scale(self, scale: float) -> None:
        if self._merged: raise RuntimeError("unmerge before changing LoRA scale")
        if scale < 0: raise ValueError("scale must be nonnegative")
        self.lora_scale.fill_(scale)

    @torch.no_grad()
    def merge_lora_(self) -> None:
        if self._merged: return
        for name in self.targets:
            base, delta = getattr(self, name), self._delta(name).to(getattr(self, name).dtype)
            base.add_(delta.reshape_as(base) if name != "decoder_down" else delta.squeeze(0))
        self._merged = True

    @torch.no_grad()
    def unmerge_lora_(self) -> None:
        if not self._merged: return
        for name in self.targets:
            base, delta = getattr(self, name), self._delta(name).to(getattr(self, name).dtype)
            base.sub_(delta.reshape_as(base) if name != "decoder_down" else delta.squeeze(0))
        self._merged = False

    def freeze_base_parameters(self) -> None:
        for name, p in self.named_parameters(): p.requires_grad_(name.startswith("lora_"))

    def adapter_parameter_count(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters() if n.startswith("lora_"))

    def load_base_state_dict(self, state_dict: dict[str, torch.Tensor], *, strict: bool = True) -> None:
        result = self.load_state_dict(state_dict, strict=False)
        bad_missing = [x for x in result.missing_keys if not (x.startswith("lora_") or x == "lora_scale")]
        bad_unexpected = [x for x in result.unexpected_keys if not x.startswith("lora_")]
        if strict and (bad_missing or bad_unexpected):
            raise RuntimeError(f"incompatible base state: missing={bad_missing}, unexpected={bad_unexpected}")
