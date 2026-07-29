"""Evaluate every preserved milestone checkpoint in a run-dir on the FULL
deterministic validation set, and snapshot whichever one scores best as
native_metal_checkpoint_best_full_holdout.

Deliberately a separate, standalone process rather than work done inside
the live trainer -- full-holdout evaluation (thousands of sequences) is
far more expensive than the trainer's own fixed-subset validation check,
and running it as part of the hot loop would compete with training for
the same GPU. Run this after milestone checkpoints are saved (or at the
end of a run), not concurrently with active training on a one-GPU
machine.

Reads --dim/--layers/--heads/--d-ff/--architecture from the run's own
config_snapshot.json when available (written by hz0a_native_stage_runner
since the A7 closure), falling back to CLI args for older runs that
predate it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from reference.hz0a_mlx_model import HZ0AMlxModel


def load_config(run_dir: Path, args) -> dict:
    snapshot_path = run_dir / "config_snapshot.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        return {
            "dim": snapshot["dim"], "layers": snapshot["layers"], "heads": snapshot["heads"],
            "d_ff": snapshot["d_ff"], "architecture": snapshot["architecture"], "vocab_size": snapshot["vocab_size"],
        }
    return {"dim": args.dim, "layers": args.layers, "heads": args.heads, "d_ff": args.d_ff, "architecture": args.architecture, "vocab_size": args.vocab_size}


def load_model_from_checkpoint(checkpoint_dir: Path, config: dict):
    payload = json.loads((checkpoint_dir / "state.json").read_text())
    attention = tuple(range(config["layers"])) if config["architecture"] == "transformer" else tuple(index for index in (4, 9, 14, 19, 24, 29) if index < config["layers"])
    model = HZ0AMlxModel(config["vocab_size"], config["dim"], config["layers"], config["heads"], config["d_ff"], attention, native_metal=True)
    model_arrays = [(item["key"], mx.load(str(checkpoint_dir / item["file"]))) for item in payload["arrays"] if item["group"] == "model"]
    model.update(tree_unflatten(model_arrays))
    mx.eval(model.parameters())
    return model, payload["step"], payload["tokens_seen"]


def full_holdout_loss(model, path: Path, batch_size: int = 4) -> tuple[float, int, int]:
    sequences = [json.loads(line) for line in path.open()]
    total_loss, total_batches = 0.0, 0
    for start in range(0, len(sequences) - batch_size + 1, batch_size):
        batch = mx.array(sequences[start:start + batch_size], dtype=mx.int32)
        logits, _ = model(batch)
        loss = mx.mean(nn.losses.cross_entropy(logits[:, :-1], batch[:, 1:]))
        mx.eval(loss)
        total_loss += float(loss)
        total_batches += 1
    return total_loss / total_batches, total_batches * batch_size, len(sequences)


def find_checkpoints(run_dir: Path) -> list[Path]:
    candidates = [run_dir / "native_metal_checkpoint"]
    candidates.extend(sorted(run_dir.glob("native_metal_checkpoint_milestone_*")))
    return [path for path in candidates if (path / "state.json").exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/repro_256_val.jsonl"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=31)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=2304)
    parser.add_argument("--vocab-size", type=int, default=24576)
    parser.add_argument("--architecture", choices=("hybrid", "transformer"), default="hybrid")
    args = parser.parse_args()

    config = load_config(args.run_dir, args)
    checkpoints = find_checkpoints(args.run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints with state.json found under {args.run_dir}")

    results = []
    for checkpoint_dir in checkpoints:
        model, step, tokens_seen = load_model_from_checkpoint(checkpoint_dir, config)
        loss, evaluated, total = full_holdout_loss(model, args.validation_data, args.batch_size)
        results.append({"checkpoint": str(checkpoint_dir), "step": step, "tokens_seen": tokens_seen, "full_holdout_val_loss": loss, "sequences_evaluated": evaluated, "sequences_total": total})
        print(f"{checkpoint_dir.name}: step={step} tokens_seen={tokens_seen} full_holdout_val_loss={loss:.6f}", flush=True)
        del model

    best = min(results, key=lambda item: item["full_holdout_val_loss"])
    best_dir = args.run_dir / "native_metal_checkpoint_best_full_holdout"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(Path(best["checkpoint"]), best_dir)

    report = {"run_dir": str(args.run_dir), "config": config, "results": results, "best": best, "best_full_holdout_checkpoint": str(best_dir)}
    (args.run_dir / "full_holdout_sweep.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nBest: {best['checkpoint']} (loss {best['full_holdout_val_loss']:.6f}) -> snapshotted to {best_dir}")


if __name__ == "__main__":
    main()
