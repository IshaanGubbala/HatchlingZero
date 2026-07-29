"""Audit an MLX-native HZ-0A stage checkpoint (native_metal_checkpoint/state.json + .npy)
without loading it onto the accelerator.

`scripts/hz0a_audit_stage_checkpoint.py` only understands the older PyTorch
`.pt` checkpoint format (`torch.load`, `payload["model"]`,
`payload["dataset_cursor"]`). `scripts/hz0a_native_stage_runner.py` -- the
harness actually in use -- writes a different, MLX-native layout: a
`state.json` manifest plus one `.npy` file per parameter/optimizer-state
leaf. This script audits that format directly with numpy, matching the A7
plan requirement for "a checkpoint audit command."
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(path: Path, required_tokens: int | None) -> dict:
    payload = json.loads((path / "state.json").read_text(encoding="utf-8"))
    metrics = payload.get("metrics", [])
    validation = [item for item in metrics if item.get("validation_loss") is not None]

    model_arrays: list[tuple[str, np.ndarray]] = []
    optimizer_leaf_count = 0
    all_finite = True
    for item in payload["arrays"]:
        array = np.load(path / item["file"])
        if not np.isfinite(array).all():
            all_finite = False
        if item["group"] == "model":
            model_arrays.append((item["key"], array))
        else:
            optimizer_leaf_count += 1

    parameter_sha256 = hashlib.sha256(b"".join(value.tobytes() for _, value in model_arrays)).hexdigest()
    tokens_seen = int(payload["tokens_seen"])

    result = {
        "checkpoint": str(path),
        "step": int(payload["step"]),
        "microbatch_count": int(payload.get("microbatch_count", 0)),
        "epoch_or_data_pass": int(payload.get("epoch_or_data_pass", 0)),
        "batch_index": int(payload["batch_index"]),
        "tokens_seen": tokens_seen,
        "metrics": len(metrics),
        "last_validation": validation[-1] if validation else None,
        "model_parameter_count": len(model_arrays),
        "optimizer_state_leaf_count": optimizer_leaf_count,
        "model_parameter_sha256": parameter_sha256,
        "checkpoint_tensors_finite": all_finite,
    }
    if required_tokens is not None:
        result["required_tokens"] = required_tokens
        result["budget_fraction"] = tokens_seen / required_tokens
        result["budget_complete"] = tokens_seen >= required_tokens
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a native_metal_checkpoint directory")
    parser.add_argument("--required-tokens", type=int, default=None, help="Optional token budget to report completion fraction against")
    args = parser.parse_args()
    if not (args.checkpoint / "state.json").exists():
        raise FileNotFoundError(f"{args.checkpoint} does not look like an MLX-native checkpoint (no state.json)")
    print(json.dumps(audit(args.checkpoint, args.required_tokens), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
