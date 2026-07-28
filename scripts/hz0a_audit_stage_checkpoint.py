"""Audit an HZ-0A stage checkpoint without loading it onto the accelerator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def audit(path: Path, required_tokens: int, batch_size: int, sequence_length: int) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metrics = payload.get("metrics", [])
    state = payload.get("dataset_cursor", {})
    tokens_seen = int(payload["step"]) * batch_size * (sequence_length - 1)
    validation = [item for item in metrics if item.get("validation_loss") is not None]
    finite = all(torch.isfinite(value).all().item() for value in payload["model"].values() if torch.is_tensor(value))
    return {
        "checkpoint": str(path),
        "step": int(payload["step"]),
        "tokens_seen": tokens_seen,
        "required_tokens": required_tokens,
        "budget_fraction": tokens_seen / required_tokens,
        "budget_complete": tokens_seen >= required_tokens,
        "metrics": len(metrics),
        "last_validation": validation[-1] if validation else None,
        "model_parameter_sha256": payload.get("model_parameter_sha256"),
        "model_tensors_finite": finite,
        "dataset_cursor": state,
        "device": payload.get("device"),
        "dtype": payload.get("dtype"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--required-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.checkpoint, args.required_tokens, args.batch_size, args.sequence_length), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
