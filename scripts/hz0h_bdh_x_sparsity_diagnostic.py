#!/usr/bin/env python3
"""Real x-sparsity (and y, g=x*y) density measurement on our ACTUAL
production checkpoint (n_embd=2496, mult=16 -- n/d=16), not the paper's
different scaling geometry (d=256, n/d~128-4096). Answers Phase 0 of the
exact-sparse-state proposal: is our regime anywhere near the paper's
claimed ~5% activity, or does our much-wider-d/narrower-n/d geometry
produce meaningfully different sparsity? Measures per-layer, per-round
density directly on a real trained checkpoint, no training needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig
from scripts.hz0h_factorized_curriculum_full_comparison import read_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device)
    config = BDHConfig(**payload["config"])
    model = BDH(config).to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    nh = config.n_head
    D = config.n_embd
    N = D * config.mlp_internal_dim_multiplier // nh

    per_round_x_density = [[] for _ in range(config.n_layer)]
    per_round_y_density = [[] for _ in range(config.n_layer)]
    per_round_g_density = [[] for _ in range(config.n_layer)]

    epochs = [0]
    with args.validation_data.open() as handle, torch.no_grad():
        for _ in range(args.eval_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx = data[:, :-1].contiguous()
            B, T = idx.shape

            x = model.ln(model.embed(idx).unsqueeze(1))
            for level in range(config.n_layer):
                x_latent = x @ model._w(model.encoder)
                x_sparse = F.relu(x_latent)
                per_round_x_density[level].append(float((x_sparse != 0).float().mean()))

                yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
                yKV = model.ln(yKV)
                y_latent = yKV @ model._w(model.encoder_v)
                y_sparse = F.relu(y_latent)
                per_round_y_density[level].append(float((y_sparse != 0).float().mean()))

                g = x_sparse * y_sparse
                per_round_g_density[level].append(float((g != 0).float().mean()))

                xy_sparse = model.drop(g)
                yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
                y = model.ln(yMLP)
                x = model.ln(x + y)

    def avg(rows):
        return [sum(r) / len(r) for r in rows]

    x_density = avg(per_round_x_density)
    y_density = avg(per_round_y_density)
    g_density = avg(per_round_g_density)

    report = {
        "checkpoint": str(args.checkpoint),
        "n_embd": D,
        "n_head": nh,
        "N_per_head": N,
        "n_over_d_ratio": (nh * N) / D,
        "paper_geometry_note": "paper's scaling runs hold d=256 while n grows to 32768-1048576, "
                                "n/d ~128-4096; our production model has n/d~16 -- a fundamentally "
                                "different, much narrower-neuron-pool regime",
        "n_layer": config.n_layer,
        "eval_batches": args.eval_batches,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "x_density_by_round": x_density,
        "y_density_by_round": y_density,
        "g_density_by_round": g_density,
        "x_density_mean": sum(x_density) / len(x_density),
        "y_density_mean": sum(y_density) / len(y_density),
        "g_density_mean": sum(g_density) / len(g_density),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
