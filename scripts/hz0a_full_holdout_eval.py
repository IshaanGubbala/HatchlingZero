"""Evaluate a native_stage_runner checkpoint on the FULL deterministic validation set.

hz0a_native_stage_runner.py's live validation_loss metric evaluates a single
rotating 1024-token sequence per checkpoint (batch_size=1, cursor advances
each call) -- useful as a cheap in-loop signal, but too high-variance to
trust for a final architecture comparison. This script re-evaluates a saved
checkpoint's final weights against every sequence in a packed validation
file, deterministically, for an honest single number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from reference.hz0a_mlx_model import HZ0AMlxModel


def load_model(run_dir: Path, dim: int, layers: int, heads: int, d_ff: int, attention: tuple[int, ...]) -> tuple:
    payload = json.loads((run_dir / "native_metal_checkpoint" / "state.json").read_text())
    model = HZ0AMlxModel(24576, dim, layers, heads, d_ff, attention, native_metal=True)
    model_arrays = [
        (item["key"], mx.load(str(run_dir / "native_metal_checkpoint" / item["file"])))
        for item in payload["arrays"] if item["group"] == "model"
    ]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-run-dir", type=Path, default=Path("outputs/hz0a_stage1_10m_native"))
    parser.add_argument("--transformer-run-dir", type=Path, default=Path("outputs/hz0a_stage1_10m_transformer"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/repro_1024_val.jsonl"))
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=31)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--hybrid-d-ff", type=int, default=2304)
    parser.add_argument("--transformer-d-ff", type=int, default=2944)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("outputs/stage1_full_holdout_comparison.json"))
    args = parser.parse_args()

    hybrid, h_step, h_tokens = load_model(args.hybrid_run_dir, args.dim, args.layers, args.heads, args.hybrid_d_ff, (4, 9, 14, 19, 24, 29))
    transformer, t_step, t_tokens = load_model(args.transformer_run_dir, args.dim, args.layers, args.heads, args.transformer_d_ff, tuple(range(args.layers)))

    h_loss, h_n, h_total = full_holdout_loss(hybrid, args.validation_data, args.batch_size)
    t_loss, t_n, t_total = full_holdout_loss(transformer, args.validation_data, args.batch_size)

    result = {
        "eval_set": str(args.validation_data) + " (full, deterministic, all sequences)",
        "hybrid": {"step": h_step, "tokens_seen": h_tokens, "full_holdout_val_loss": h_loss, "sequences_evaluated": h_n, "sequences_total": h_total},
        "transformer": {"step": t_step, "tokens_seen": t_tokens, "full_holdout_val_loss": t_loss, "sequences_evaluated": t_n, "sequences_total": t_total},
        "gap_hybrid_minus_transformer": h_loss - t_loss,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
