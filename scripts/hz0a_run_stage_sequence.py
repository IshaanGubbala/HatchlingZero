"""Run the declared HZ-0A stages in order with resumable per-stage reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--validation-data", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--stage-config", type=Path, default=Path("configs/hz0a_training_stages.json"))
    parser.add_argument("--model-config", type=Path, default=Path("specs/hz0a_300m_a1.json"))
    parser.add_argument("--transformer-config", type=Path, default=Path("configs/hz0a_transformer_matched.json"))
    parser.add_argument("--models", default="locked,matched_transformer")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=24576)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.dtype == "fp16" and args.device != "mps":
        raise ValueError("fp16 sequence runs require MPS")
    stages = json.loads(args.stage_config.read_text(encoding="utf-8"))["stages"]
    args.run_root.mkdir(parents=True, exist_ok=True)
    results = []
    for stage in stages:
        stage_dir = args.run_root / stage["name"]
        command = [sys.executable, "scripts/hz0a_stage_runner.py", "--stage-config", str(args.stage_config), "--stage", stage["name"], "--data", str(args.data), "--validation-data", str(args.validation_data), "--run-dir", str(stage_dir), "--batch-size", str(args.batch_size), "--vocab-size", str(args.vocab_size), "--checkpoint-interval", str(args.checkpoint_interval), "--validation-interval", str(args.validation_interval), "--device", args.device, "--dtype", args.dtype, "--models", args.models, "--model-config", str(args.model_config), "--transformer-config", str(args.transformer_config)]
        if args.resume:
            command.append("--resume")
        completed = subprocess.run(command, check=False)
        report_path = stage_dir / "stage_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {"stage": stage["name"], "returncode": completed.returncode}
        results.append(report)
        if completed.returncode != 0 or not all(item.get("budget_complete", False) for item in report.get("models", {}).values()):
            raise SystemExit(f"stage {stage['name']} did not complete for every requested model")
    manifest = {"stage_config": str(args.stage_config), "models": args.models.split(","), "stages": [item.get("stage") for item in results], "reports": results}
    (args.run_root / "sequence_report.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stages_completed": manifest["stages"], "sequence_report": str(args.run_root / "sequence_report.json")}, indent=2))


if __name__ == "__main__":
    main()
