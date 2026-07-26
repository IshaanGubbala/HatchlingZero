from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hz0.model import HybridLM
from hz0.model.backends import gdn2_is_available, gdn2_status


def test_fallback_model_forward() -> None:
    model = HybridLM(
        vocab_size=256,
        d_model=64,
        n_layers=4,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        mixer_backend="fallback",
        attention_every=2,
        max_seq_len=128,
    )
    x = torch.randint(0, 256, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 256)


def test_auto_backend_forward_or_fallback() -> None:
    available, _ = gdn2_is_available()
    model = HybridLM(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        mixer_backend="auto",
        attention_every=2,
        max_seq_len=64,
    )
    x = torch.randint(0, 256, (1, 8))
    logits = model(x)
    assert logits.shape == (1, 8, 256)
    assert available in (True, False)


def test_gdn2_status_shape() -> None:
    status = gdn2_status()
    assert "available" in status
    assert "reason" in status
