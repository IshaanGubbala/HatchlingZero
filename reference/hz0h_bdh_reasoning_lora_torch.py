"""Training-only reasoning LoRA for the faithful BDH oracle.

This module deliberately leaves ``reference.hz0h_bdh_torch.BDH`` unchanged.
It adds zero-initialized low-rank deltas to BDH's value/write paths
(``encoder_v`` and ``decoder``) while leaving the sensitive addressing
projection (``encoder``) off by default.  At construction the effective model
is bit-identical to BDH; adapters can be annealed to scale zero, or merged
into the base weights for ordinary LoRA deployment.

It is plumbing for the A/B/C/D training-only-capacity experiment, not evidence
that reasoning transfers after adapter removal.  In particular, setting scale
zero is intentionally an exact base-only evaluation, whereas merging retains
the learned adapter effect in the base weights.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

from reference.hz0h_bdh_torch import BDH, BDHConfig

_VALUE_TARGETS = frozenset(("encoder_v", "decoder"))
_ALL_TARGETS = _VALUE_TARGETS | frozenset(("encoder",))


class ReasoningLoRABDH(BDH):
    """BDH with optional low-rank deltas on named shared projections.

    ``rank`` is per head: encoder_v's delta is ``A[D,r] @ B[r,N]`` and
    decoder's is ``A[N,r] @ B[r,D]``.  B begins at zero, so adapters do not
    perturb a loaded or freshly constructed base model before training.
    """

    def __init__(
        self,
        config: BDHConfig,
        *,
        rank: int = 8,
        alpha: float | None = None,
        targets: Iterable[str] = ("encoder_v", "decoder"),
        freeze_base: bool = False,
    ) -> None:
        if rank < 1:
            raise ValueError("rank must be positive")
        targets = tuple(targets)
        unknown = set(targets) - _ALL_TARGETS
        if unknown:
            raise ValueError(f"unknown LoRA target(s): {sorted(unknown)}; allowed: {sorted(_ALL_TARGETS)}")
        if not targets:
            raise ValueError("at least one LoRA target is required")
        super().__init__(config)
        self.rank = rank
        self.lora_alpha = float(rank if alpha is None else alpha)
        if self.lora_alpha <= 0:
            raise ValueError("alpha must be positive")
        self.register_buffer("lora_scale", torch.tensor(1.0), persistent=True)
        self.targets = frozenset(targets)
        self._merged = False

        nh, D = config.n_head, config.n_embd
        N = D * config.mlp_internal_dim_multiplier // nh
        if "encoder" in self.targets:
            self.lora_encoder_A = nn.Parameter(torch.empty(nh, D, rank))
            self.lora_encoder_B = nn.Parameter(torch.zeros(nh, rank, N))
        if "encoder_v" in self.targets:
            self.lora_encoder_v_A = nn.Parameter(torch.empty(nh, D, rank))
            self.lora_encoder_v_B = nn.Parameter(torch.zeros(nh, rank, N))
        if "decoder" in self.targets:
            self.lora_decoder_A = nn.Parameter(torch.empty(nh, N, rank))
            self.lora_decoder_B = nn.Parameter(torch.zeros(nh, rank, D))
        self.reset_lora_parameters()
        if freeze_base:
            self.freeze_base_parameters()

    @property
    def adapter_multiplier(self) -> torch.Tensor:
        return self.lora_scale * (self.lora_alpha / self.rank)

    def reset_lora_parameters(self) -> None:
        """Reset to exact-base behavior while retaining trainable A factors."""
        for target in self.targets:
            a = getattr(self, f"lora_{target}_A")
            b = getattr(self, f"lora_{target}_B")
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            nn.init.zeros_(b)

    def lora_delta(self, target: str) -> torch.Tensor:
        if target not in self.targets:
            raise KeyError(f"{target!r} is not an enabled LoRA target")
        return torch.matmul(
            getattr(self, f"lora_{target}_A"),
            getattr(self, f"lora_{target}_B"),
        ) * self.adapter_multiplier.to(dtype=getattr(self, target).dtype)

    def _w(self, param: torch.Tensor) -> torch.Tensor:
        # Preserve BDH's ternary hook on the base weight, then add the
        # full-precision training adapter. Identity checks avoid fragile name
        # dispatch and preserve behavior for callers passing unrelated tensors.
        base = super()._w(param)
        if self._merged or float(self.lora_scale.detach()) == 0.0:
            return base
        for target in self.targets:
            if param is getattr(self, target):
                delta = self.lora_delta(target)
                return base + (delta.reshape_as(param) if target == "decoder" else delta)
        return base

    @torch.no_grad()
    def set_lora_scale(self, scale: float) -> None:
        """Set a nonnegative annealing coefficient; zero is exact base-only."""
        if self._merged:
            raise RuntimeError("cannot scale a merged adapter; unmerge it first")
        if scale < 0:
            raise ValueError("scale must be nonnegative")
        self.lora_scale.fill_(scale)

    @torch.no_grad()
    def merge_lora_(self) -> None:
        """Fold the current scaled delta into base weights, removing runtime adds.

        The adapter parameters remain in the state dict for reversibility; use
        ``unmerge_lora_`` before changing their scale or training them again.
        """
        if self._merged:
            return
        for target in self.targets:
            param = getattr(self, target)
            delta = self.lora_delta(target)
            param.add_(delta.reshape_as(param) if target == "decoder" else delta)
        self._merged = True

    @torch.no_grad()
    def unmerge_lora_(self) -> None:
        if not self._merged:
            return
        for target in self.targets:
            param = getattr(self, target)
            delta = self.lora_delta(target)
            param.sub_(delta.reshape_as(param) if target == "decoder" else delta)
        self._merged = False

    def freeze_base_parameters(self) -> None:
        """Freeze every non-LoRA parameter for adapter-only teacher training."""
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("lora_"))

    def unfreeze_base_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for name, parameter in self.named_parameters() if name.startswith("lora_"))

    def load_base_state_dict(self, state_dict: dict[str, torch.Tensor], *, strict: bool = True) -> None:
        """Load a plain BDH checkpoint without treating newly-created LoRA keys as errors."""
        result = self.load_state_dict(state_dict, strict=False)
        unexpected = [key for key in result.unexpected_keys if not key.startswith("lora_")]
        missing = [key for key in result.missing_keys if not (key.startswith("lora_") or key == "lora_scale")]
        if strict and (unexpected or missing):
            raise RuntimeError(f"incompatible base checkpoint; missing={missing}, unexpected={unexpected}")


def linear_lora_scale(step: int, start_step: int, end_step: int) -> float:
    """A 1→0 linear removal schedule inclusive of its endpoints."""
    if end_step <= start_step:
        raise ValueError("end_step must be greater than start_step")
    if step <= start_step:
        return 1.0
    if step >= end_step:
        return 0.0
    return 1.0 - (step - start_step) / (end_step - start_step)
