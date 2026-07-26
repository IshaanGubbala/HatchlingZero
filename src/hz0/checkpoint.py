from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    metrics: dict[str, float] | None = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"step_{step:07d}.pt"
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "metrics": metrics or {},
    }
    torch.save(payload, checkpoint_path)
    latest_path = out_dir / "latest.pt"
    torch.save(payload, latest_path)
    if metrics is not None:
        metrics_path = out_dir / f"step_{step:07d}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return checkpoint_path


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)
