"""Validate that a packed dataset is large enough for an HZ-0A stage."""

import argparse
import hashlib
import json
from pathlib import Path


def stage_gate(stage_config: Path, packed_data: Path, stage_name: str) -> dict:
    config = json.loads(stage_config.read_text(encoding="utf-8"))
    stages = {stage["name"]: stage for stage in config["stages"]}
    if stage_name not in stages:
        raise ValueError(f"unknown stage: {stage_name}")
    if packed_data.suffix == ".jsonl":
        sequence_count = 0
        available_tokens = 0
        with packed_data.open(encoding="utf-8") as reader:
            for line_number, line in enumerate(reader, 1):
                sequence = json.loads(line)
                if not isinstance(sequence, list):
                    raise ValueError(f"packed data line {line_number} is not a token sequence")
                sequence_count += 1
                available_tokens += len(sequence)
    else:
        sequences = json.loads(packed_data.read_text(encoding="utf-8"))
        if not sequences or any(not isinstance(sequence, list) for sequence in sequences):
            raise ValueError("packed data must be a non-empty list of token sequences")
        sequence_count = len(sequences)
        available_tokens = sum(len(sequence) for sequence in sequences)
    required_tokens = int(stages[stage_name]["tokens"])
    report = {
        "stage": stage_name,
        "required_tokens": required_tokens,
        "available_tokens": available_tokens,
        "sequence_count": sequence_count,
        "packed_data_sha256": hashlib.sha256(packed_data.read_bytes()).hexdigest(),
        "sufficient": available_tokens >= required_tokens,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check an HZ-0A stage data budget before launch.")
    parser.add_argument("--stage-config", default="configs/hz0a_training_stages.json", type=Path)
    parser.add_argument("--packed-data", default="data/packed/train_packed.json", type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = stage_gate(args.stage_config, args.packed_data, args.stage)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["sufficient"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
