"""Run a real or meta-device forward smoke for the locked HZ-0A model."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.hz0a_torch_model import HZ0AConfig, HZ0AModel


def run(config_path: Path, batch_size: int, sequence_length: int, meta: bool) -> dict:
    config = HZ0AConfig.from_json(config_path)
    started = time.perf_counter()
    with torch.device("meta") if meta else torch.device("cpu"):
        model = HZ0AModel(config)
        token_ids = torch.zeros((batch_size, sequence_length), dtype=torch.long, device="meta" if meta else "cpu")
        if not meta:
            with torch.no_grad():
                logits, states = model(token_ids)
            finite = bool(torch.isfinite(logits).all())
            state_shapes = [None if state is None else list(state.shape) for state in states]
        else:
            finite = None
            state_shapes = [None if block.attention else [batch_size, config.num_heads, config.d_v, config.d_k] for block in model.blocks]
    return {"config": str(config_path), "meta": meta, "batch_size": batch_size, "sequence_length": sequence_length, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "finite_logits": finite, "state_shapes": state_shapes, "elapsed_seconds": time.perf_counter() - started, "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the locked HZ-0A full model.")
    parser.add_argument("--config", default="specs/hz0a_300m_a1.json", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--meta", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.config, args.batch_size, args.sequence_length, args.meta)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
