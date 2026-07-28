"""Audit a multi-parameter tiny HZ-0A PyTorch checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def fingerprint_state_dict(state_dict: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def walk_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from walk_tensors(item)


def audit(path: Path) -> dict:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    required = {"model", "optimizer", "step", "batch_index", "metrics", "torch_rng", "initial_parameter_sha256", "model_parameter_sha256"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing checkpoint fields: {sorted(missing)}")
    step = int(payload["step"])
    metrics = payload["metrics"]
    if step != len(metrics) or [int(item["step"]) for item in metrics] != list(range(1, step + 1)):
        raise ValueError("metric step continuity is invalid")
    tensors = list(walk_tensors(payload["model"])) + list(walk_tensors(payload["optimizer"]))
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors if tensor.is_floating_point()):
        raise ValueError("checkpoint contains non-finite tensor values")
    actual_hash = fingerprint_state_dict(payload["model"])
    if actual_hash != payload["model_parameter_sha256"]:
        raise ValueError("model parameter fingerprint does not match checkpoint")
    parameter_count = sum(tensor.numel() for tensor in payload["model"].values() if isinstance(tensor, torch.Tensor))
    parameter_bytes = sum(tensor.numel() * tensor.element_size() for tensor in payload["model"].values() if isinstance(tensor, torch.Tensor))
    return {"valid": True, "step": step, "batch_index": int(payload["batch_index"]), "metric_count": len(metrics), "parameter_count": parameter_count, "parameter_bytes": parameter_bytes, "model_parameter_sha256": actual_hash, "optimizer_state_entries": len(payload["optimizer"].get("state", {}))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a tiny HZ-0A multi-parameter checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.checkpoint), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
