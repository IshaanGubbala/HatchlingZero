from __future__ import annotations

from typing import Any

from torch import nn

from .hybrid_lm import HybridLM
from .transformer_lm import TransformerLM


def build_model(config: dict[str, Any]) -> nn.Module:
    architecture = str(config.get("architecture", "hybrid")).lower()
    kwargs = dict(config)
    kwargs.pop("architecture", None)
    if architecture == "hybrid":
        return HybridLM(**kwargs)
    if architecture == "transformer":
        return TransformerLM(**kwargs)
    raise ValueError(f"Unknown model architecture: {architecture}")
