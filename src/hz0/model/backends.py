from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch
from torch import nn


class BackendUnavailableError(RuntimeError):
    pass


def _vendor_repo() -> Path:
    return Path(__file__).resolve().parents[3] / "vendor" / "GatedDeltaNet-2"


def _vendor_lit_gpt_dir() -> Path:
    return _vendor_repo() / "lit_gpt"


def _load_vendor_gdn2_module():
    lit_gpt_dir = _vendor_lit_gpt_dir()
    if not lit_gpt_dir.exists():
        raise BackendUnavailableError("Vendored lit_gpt directory not found.")

    pkg = types.ModuleType("lit_gpt")
    pkg.__path__ = [str(lit_gpt_dir)]
    sys.modules["lit_gpt"] = pkg
    return importlib.import_module("lit_gpt.gdn2")


def _apply_fla_compat_shims() -> None:
    """Patch small FLA API drifts used by the vendored GDN-2 code.

    The experimental triton-msl branch currently uses a newer installable FLA
    build where `USE_CUDA_GRAPH` is no longer exported from `fla.utils`.
    The vendored GDN-2 kernels only use it as a boolean autotune flag, so a
    conservative `False` default is a safe compatibility bridge for imports.
    """
    try:
        import fla.utils
    except Exception:
        return
    if not hasattr(fla.utils, "USE_CUDA_GRAPH"):
        fla.utils.USE_CUDA_GRAPH = False


def gdn2_status() -> dict[str, str | bool]:
    repo = _vendor_repo()
    if not repo.exists():
        return {
            "available": False,
            "reason": "Vendored GatedDeltaNet-2 repository not found.",
        }

    try:
        import fla  # noqa: F401
    except Exception as exc:  # pragma: no cover - import depends on local system stack
        return {
            "available": False,
            "reason": f"Missing flash-linear-attention dependency: {exc}",
        }
    _apply_fla_compat_shims()

    try:
        module = _load_vendor_gdn2_module()
        getattr(module, "GatedDeltaNet2")
    except Exception as exc:  # pragma: no cover - import depends on local system stack
        return {
            "available": False,
            "reason": str(exc),
        }
    return {
        "available": True,
        "reason": "",
    }


def gdn2_is_available() -> tuple[bool, str | None]:
    status = gdn2_status()
    return bool(status["available"]), str(status["reason"]) or None


class UpstreamGDN2Mixer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        available, reason = gdn2_is_available()
        if not available:
            raise BackendUnavailableError(
                "GatedDeltaNet-2 backend is unavailable. "
                f"Reason: {reason}"
            )

        module = _load_vendor_gdn2_module()
        GatedDeltaNet2 = getattr(module, "GatedDeltaNet2")

        head_dim = d_model // n_heads
        self.inner = GatedDeltaNet2(
            hidden_size=d_model,
            num_heads=n_heads,
            head_dim=head_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _, _ = self.inner(x, attention_mask=None)
        return out
