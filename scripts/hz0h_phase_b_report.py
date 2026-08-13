"""HZ Next-Phase Plan Phase B: aggregated report across the real trained
checkpoints (exact BDH + VB at D/2, D/3, D/4, all with the locked
recurrent-depth curriculum), rather than reading each result from a
separate ad-hoc script/inline check the way HZ-Core-1's own results
were pieced together. Real question this answers: where does the
quality/memory Pareto point actually land -- does a milder compression
ratio (D/2, D/3) beat D/4's already-measured 1.6309, and by how much
relative to exact-BDH-curriculum's 1.5820 baseline?

Reuses, does not reimplement: validation-CE reproduction methodology
from the checkpoint-loading sanity check used earlier this session,
`scripts/hz0h_state_memory_analysis.py`/`_vb.py`'s real state-byte
formulas, and `scripts/hz0h_core1_checkpoint_quality_eval.py`'s
passkey/reassignment eval functions (real-text-plausible value
alphabet, not the confounded original H5 byte range).

All checkpoint arguments are optional -- pass only the ones you have on
hand; the report prints whatever real data is available and clearly
marks anything not provided as skipped, rather than failing on missing
arguments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_train_torch import shifted_target_batch
from reference.hz0h_bdh_vb_torch import BDHVB, BDHVBConfig
from scripts.hz0h_core1_checkpoint_quality_eval import (
    evaluate_bdh_passkey,
    evaluate_bdh_reassignment,
    evaluate_vb_passkey,
    evaluate_vb_reassignment,
    load_bdh,
    load_bdh_vb,
)
from scripts.hz0h_state_memory_analysis import state_bytes_per_batch_item as bdh_state_bytes
from scripts.hz0h_state_memory_analysis_vb import (
    state_bytes_per_batch_item_fp32 as vb_state_bytes_fp32,
    state_bytes_per_batch_item_int8 as vb_state_bytes_int8,
)


@torch.no_grad()
def real_validation_loss(model, *, validation_data: Path, n_embd: int, n_layer: int, n_head: int, mlp_internal_dim_multiplier: int, vocab_size: int, sequence_length: int = 256, num_sequences: int = 64, is_vb: bool) -> float:
    lines = []
    with validation_data.open() as f:
        for i, line in enumerate(f):
            if i >= num_sequences:
                break
            lines.append(json.loads(line)[:sequence_length])
    batch = torch.tensor(lines, dtype=torch.long)
    x, y = shifted_target_batch(batch)
    _logits, loss = model(x, targets=y)
    return float(loss)


def report_row(label: str, params: int, d_state: int | None, val_loss: float, state_bytes_fp32: int, state_bytes_int8: int | None, passkey_real: float | None, reassignment_real: float | None) -> dict:
    return {
        "label": label, "parameter_count": params, "d_state": d_state,
        "validation_loss": val_loss,
        "state_bytes_per_batch_item_fp32": state_bytes_fp32,
        "state_bytes_per_batch_item_int8": state_bytes_int8,
        "passkey_real_state_accuracy": passkey_real,
        "reassignment_real_state_accuracy": reassignment_real,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--num-eval-examples", type=int, default=200)
    parser.add_argument("--skip-memory-tasks", action="store_true", help="skip passkey/reassignment (real but slower on CPU/token-by-token streaming) -- validation CE and state memory still computed")
    parser.add_argument("--exact-bdh-checkpoint", type=Path, default=None)
    parser.add_argument("--vb-d2-checkpoint", type=Path, default=None)
    parser.add_argument("--vb-d3-checkpoint", type=Path, default=None)
    parser.add_argument("--vb-d4-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    common = dict(n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head, mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size)
    rows = []

    if args.exact_bdh_checkpoint is not None:
        model = load_bdh(args.exact_bdh_checkpoint, **common)
        params = sum(p.numel() for p in model.parameters())
        val_loss = real_validation_loss(model, validation_data=args.validation_data, **common, is_vb=False)
        state_bytes = bdh_state_bytes(BDHConfig(**common, dropout=0.0))
        passkey_real = reassignment_real = None
        if not args.skip_memory_tasks:
            passkey_real = evaluate_bdh_passkey(model, prefix_len=4, filler_len=16, value_range=8, num_examples=args.num_eval_examples, seed=1000)["real_state_accuracy"]
            reassignment_real = evaluate_bdh_reassignment(model, prefix_len=4, filler_len=8, value_range=8, num_reassignments=3, num_examples=args.num_eval_examples, seed=2000)["real_state_accuracy"]
        rows.append(report_row("Exact BDH (curriculum)", params, None, val_loss, state_bytes, None, passkey_real, reassignment_real))

    for label, ckpt, d_state in (
        ("VB D/2 (curriculum)", args.vb_d2_checkpoint, args.n_embd // 2),
        ("VB D/3 (curriculum)", args.vb_d3_checkpoint, round(args.n_embd / 3)),
        ("VB D/4 (curriculum)", args.vb_d4_checkpoint, args.n_embd // 4),
    ):
        if ckpt is None:
            continue
        model = load_bdh_vb(ckpt, **common, d_state=d_state)
        params = sum(p.numel() for p in model.parameters())
        val_loss = real_validation_loss(model, validation_data=args.validation_data, **common, is_vb=True)
        vb_config = BDHVBConfig(**common, dropout=0.0, d_state=d_state)
        state_bytes_fp32 = vb_state_bytes_fp32(vb_config)
        state_bytes_int8 = vb_state_bytes_int8(vb_config)
        passkey_real = reassignment_real = None
        if not args.skip_memory_tasks:
            passkey_real = evaluate_vb_passkey(model, prefix_len=4, filler_len=16, value_range=8, num_examples=args.num_eval_examples, seed=1000, int8=False)["real_state_accuracy"]
            reassignment_real = evaluate_vb_reassignment(model, prefix_len=4, filler_len=8, value_range=8, num_reassignments=3, num_examples=args.num_eval_examples, seed=2000, int8=False)["real_state_accuracy"]
        rows.append(report_row(label, params, d_state, val_loss, state_bytes_fp32, state_bytes_int8, passkey_real, reassignment_real))

    if not rows:
        print("No checkpoints provided -- nothing to report. Pass at least one of --exact-bdh-checkpoint / --vb-d2-checkpoint / --vb-d3-checkpoint / --vb-d4-checkpoint.")
        return

    baseline_loss = rows[0]["validation_loss"]
    for row in rows:
        row["relative_to_first_row"] = f"{(row['validation_loss'] - baseline_loss) / baseline_loss * 100:+.2f}%"

    print(json.dumps(rows, indent=2))

    print("\n=== Summary ===")
    header = f"{'Label':<24} {'d_state':>8} {'ValLoss':>9} {'vs first':>10} {'StateB(fp32)':>13} {'Passkey':>8} {'Reassign':>9}"
    print(header)
    for row in rows:
        d_state_str = str(row["d_state"]) if row["d_state"] is not None else "-"
        passkey_str = f"{row['passkey_real_state_accuracy']:.3f}" if row["passkey_real_state_accuracy"] is not None else "-"
        reassign_str = f"{row['reassignment_real_state_accuracy']:.3f}" if row["reassignment_real_state_accuracy"] is not None else "-"
        print(f"{row['label']:<24} {d_state_str:>8} {row['validation_loss']:>9.4f} {row['relative_to_first_row']:>10} {row['state_bytes_per_batch_item_fp32']:>13,} {passkey_str:>8} {reassign_str:>9}")


if __name__ == "__main__":
    main()
