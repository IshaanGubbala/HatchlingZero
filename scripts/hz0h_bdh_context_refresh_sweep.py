#!/usr/bin/env python3
"""Load a trained BDH checkpoint (scripts/hz0h_bdh_checkpoint_train_for_ablation.py)
and sweep how often recurrent reasoning actually needs to re-query the
persistent state: refresh_every in {1, 2, 4, 8} (1 = vanilla BDH's real
behavior, 8 = read once at round 0 and reuse it for all 8 rounds).
Inference only, no retraining -- real val_loss for each setting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_context_refresh_ablation_torch import bdh_context_refresh_forward
from reference.hz0h_bdh_torch import BDH, BDHConfig
from scripts.hz0h_bdh_combined_best_comparison import autocast_context
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device
from scripts.hz0h_factorized_curriculum_full_comparison import read_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--refresh-intervals", type=str, default="1,2,4,8")
    args = parser.parse_args()

    device = pick_device(args.device)
    payload = torch.load(args.checkpoint, map_location=device)
    config = BDHConfig(**payload["config"])
    model = BDH(config).to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    results = {}
    for refresh_every in [int(x) for x in args.refresh_intervals.split(",")]:
        epochs = [0]
        losses = []
        real_reads_seen = None
        with args.validation_data.open() as handle, torch.no_grad(), autocast_context(args, device):
            for _ in range(args.eval_batches):
                data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
                idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
                _, loss, real_reads = bdh_context_refresh_forward(
                    model, idx, config.n_layer, refresh_every, target,
                )
                losses.append(float(loss))
                real_reads_seen = real_reads
        val_loss = sum(losses) / len(losses)
        print(f"[refresh_every={refresh_every}] real_state_reads={real_reads_seen}/{config.n_layer} "
              f"validation_loss={val_loss:.4f}", flush=True)
        results[refresh_every] = {"validation_loss": val_loss, "real_state_reads": real_reads_seen}

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_final_train_loss": payload.get("final_train_loss"),
        "checkpoint_target_tokens": payload.get("target_tokens"),
        "n_layer": config.n_layer,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
