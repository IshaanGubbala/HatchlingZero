#!/usr/bin/env python3
"""Evaluate a saved exact-BDH checkpoint's validation loss, same
methodology (batch=8/seq=256, 8 eval batches, bf16 autocast) as every
quality-check script tonight, so the number is directly comparable to
the seed=7 baseline (1.8585) and the subspace-decoder results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_wide_gemm_checkpointed_torch import bdh_wide_gemm_forward_checkpointed
from scripts.hz0h_bdh_combined_best_comparison import autocast_context
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device
from scripts.hz0h_factorized_curriculum_full_comparison import read_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = BDHConfig(**ckpt["config"])
    model = BDH(config).to(device=device, dtype=torch.float32)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    epochs = [0]
    losses = []
    with args.validation_data.open() as handle, torch.no_grad(), autocast_context(args, device):
        for _ in range(args.eval_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            _, loss = bdh_wide_gemm_forward_checkpointed(model, idx, config.n_layer, target)
            losses.append(float(loss))
    val_loss = sum(losses) / len(losses)
    print(f"[checkpoint_eval] checkpoint={args.checkpoint} validation_loss={val_loss}", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "checkpoint": str(args.checkpoint),
            "seed": ckpt.get("seed"),
            "validation_loss": val_loss,
        }, indent=2), encoding="utf-8")
        print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
