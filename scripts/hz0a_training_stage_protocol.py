#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_STAGES = ["stage1_validation", "stage2_pilot", "stage3_architecture_pilot", "stage4_full_comparison"]


def validate_protocol(protocol: dict) -> dict:
    stages = protocol["stages"]
    names = [stage["name"] for stage in stages]
    if names != REQUIRED_STAGES:
        raise ValueError(f"stage order must be {REQUIRED_STAGES}")
    if protocol["models"] != ["hz0a_300m", "hz0a_transformer_matched"]:
        raise ValueError("hybrid and matched transformer must share the protocol")
    for field in ("tokenizer_path", "data_manifest_path", "sequence_length", "effective_batch_tokens", "precision", "optimizer", "learning_rate"):
        if field not in protocol:
            raise ValueError(f"protocol missing shared field: {field}")
    if any(int(stage["tokens"]) <= 0 for stage in stages):
        raise ValueError("stage token budgets must be positive")
    return {
        "valid": True,
        "stage_count": len(stages),
        "total_tokens": sum(int(stage["tokens"]) for stage in stages),
        "shared_effective_batch_tokens": protocol["effective_batch_tokens"],
        "models": protocol["models"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the HZ-0A staged-training protocol.")
    parser.add_argument("--config", default="configs/hz0a_training_stages.json")
    args = parser.parse_args()
    report = validate_protocol(json.loads(Path(args.config).read_text(encoding="utf-8")))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
