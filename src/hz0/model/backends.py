from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


class BackendUnavailableError(RuntimeError):
    pass


def _vendor_repo() -> Path:
    return Path(__file__).resolve().parents[3] / "vendor" / "GatedDeltaNet-2"


def gdn2_is_available() -> tuple[bool, str | None]:
    repo = _vendor_repo()
    if not repo.exists():
        return False, "Vendored GatedDeltaNet-2 repository not found."

    sys.path.insert(0, str(repo))
    try:
        from lit_gpt.gdn2 import GatedDeltaNet2  # noqa: F401
    except Exception as exc:  # pragma: no cover - import depends on local system stack
        return False, str(exc)
    return True, None


class UpstreamGDN2Mixer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        available, reason = gdn2_is_available()
        if not available:
            raise BackendUnavailableError(
                "GatedDeltaNet-2 backend is unavailable. "
                f"Reason: {reason}"
            )

        repo = _vendor_repo()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from lit_gpt.gdn2 import GatedDeltaNet2

        head_dim = d_model // n_heads
        self.inner = GatedDeltaNet2(
            hidden_size=d_model,
            num_heads=n_heads,
            head_dim=head_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _, _ = self.inner(x, attention_mask=None)
        return out
