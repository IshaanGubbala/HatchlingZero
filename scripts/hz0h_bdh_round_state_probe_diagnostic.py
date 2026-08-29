#!/usr/bin/env python3
"""Phase 1 of the "internal computation" architecture phase (proposed
2026-08-29): does BDH's recurrent depth (weight-tied rounds, R=n_layer)
build up PROGRESSIVELY more decodable task-relevant state as r
increases? This is the premise the whole "teach the state we already
pay for to think" research direction rests on -- if z_2 and z_8 carry
essentially identical task-relevant information, the recurrence-as-
internal-reasoning premise needs revision before anything else in that
ladder (round embeddings, state supervision, inference-time compute
scaling) is worth building.

Real, standard probing-classifier methodology (Alain & Bengio 2016
style: freeze the model, train small linear/MLP probes on its
activations, measure probe accuracy), applied across BDH's recurrent-
round axis instead of a Transformer's independent-layer axis --
BDH's rounds are weight-tied depth iterations over the WHOLE sequence
in parallel (full self-attention within each round), not a sequential
pass over time, so z_r is read as "the last-token representation after
r rounds of full-sequence computation," analogous to probing layer r
of a Transformer. Diagnostic-only: probes are trained on FROZEN model
activations, add zero inference parameters, discarded after this
measurement.

Task: transitive object-location tracking ("X is in A. A is moved into
B. Where is X?"), real synthetic generator, closed 8-way answer
vocabulary (byte-length-independent probe target), hop-count (number
of movements) as a real, controllable difficulty axis.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig

LOCATIONS = ["the kitchen", "the garage", "the office", "the attic",
             "the basement", "the pantry", "the closet", "the yard"]
OBJECTS = ["the key", "the ball", "the box", "the phone",
           "the book", "the hat", "the lamp", "the mug"]


def generate_example(rng: random.Random, n_hops: int) -> tuple[str, int]:
    obj = rng.choice(OBJECTS)
    chain = rng.sample(LOCATIONS, n_hops + 1)
    text = f"{obj} is in {chain[0]}. "
    for i in range(n_hops):
        text += f"{chain[i]} is moved into {chain[i + 1]}. "
    text += f"Where is {obj}?"
    answer_idx = LOCATIONS.index(chain[-1])
    return text, answer_idx


def load_model(checkpoint_path: Path, device) -> BDHVBSubspaceDecoder:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = BDHVBSubspaceDecoderConfig(**ckpt["config"])
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.float32)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[probe] loaded checkpoint {checkpoint_path} n_layer={config.n_layer}", flush=True)
    return model


@torch.no_grad()
def collect_all_round_last_token(model: BDHVBSubspaceDecoder, idx: torch.Tensor, n_rounds: int | None = None) -> list[torch.Tensor]:
    """Real, exact (non-checkpointed) forward, mirroring
    BDHVBSubspaceDecoder.forward's own round loop bit-for-bit, but
    additionally returning the LAST-TOKEN residual-stream state after
    EVERY round (not just the final one). n_rounds defaults to the
    model's own trained config.n_layer; passing a LARGER value runs
    the SAME weight-tied round computation beyond its trained depth --
    architecturally valid (weights are shared/reused every round by
    construction, not per-round-specific), the real mechanism the
    R-scaling baseline is testing."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh
    n_rounds = n_rounds if n_rounds is not None else C.n_layer

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    per_round_last_token = []
    for _level in range(n_rounds):
        x_latent = x @ model.encoder
        x_sparse = F.relu(x_latent)
        v_bottleneck = x @ model.P
        yKV_bottleneck = model.attn(Q=x_sparse, K=x_sparse, V=v_bottleneck)
        yKV = yKV_bottleneck @ model.O
        yKV = model.ln(yKV)
        y_latent = yKV @ model.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        alpha = torch.matmul(xy_sparse, model.decoder_up.view(nh, N, -1)).sum(dim=1, keepdim=True)
        yMLP = alpha @ model.decoder_down
        y = model.ln(yMLP)
        x = model.ln(x + y)
        per_round_last_token.append(x[:, :, -1, :].reshape(B, D).detach().cpu())
    return per_round_last_token


def verify_against_real_forward(model: BDHVBSubspaceDecoder, idx: torch.Tensor) -> None:
    """Real sanity check before trusting anything: the LAST round's
    collected last-token state, run through lm_head, must reproduce
    the model's own real forward() logits at the last position
    exactly (up to float tolerance)."""
    per_round = collect_all_round_last_token(model, idx)
    manual_logits = per_round[-1].to(idx.device) @ model.lm_head
    with torch.no_grad():
        real_logits, _ = model(idx)
    real_last = real_logits[:, -1, :]
    diff = (manual_logits - real_last).abs().max().item()
    assert diff < 1e-3, f"collection function diverges from real forward: max diff {diff}"
    print(f"[probe] verified against real forward, max logit diff={diff:.2e}", flush=True)


class LinearProbe(nn.Module):
    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_in, n_classes)

    def forward(self, x):
        return self.linear(x)


def train_and_eval_probe(train_z: torch.Tensor, train_y: torch.Tensor, test_z: torch.Tensor, test_y: torch.Tensor,
                          n_classes: int, epochs: int, lr: float) -> float:
    probe = LinearProbe(train_z.shape[-1], n_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(train_z)
        loss = F.cross_entropy(logits, train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        test_acc = (probe(test_z).argmax(dim=-1) == test_y).float().mean().item()
    return test_acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/local/hz0h_vb_subspace_decoder_50m_500mtok.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-hops", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--probe-epochs", type=int, default=300)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-round", type=int, default=None,
                         help="R-scaling baseline: run the weight-tied recurrence for MORE rounds than "
                              "the model's trained config.n_layer (architecturally valid -- weights are "
                              "shared/reused every round, not per-round-specific). Defaults to config.n_layer "
                              "(Phase 1's original in-training-depth-only measurement).")
    args = parser.parse_args()

    from scripts.hz0h_bdh_width_flop_frontier_local import pick_device
    device = pick_device(args.device)
    model = load_model(args.checkpoint, device)
    rng = random.Random(args.seed)

    # Real sanity check before trusting anything downstream.
    verify_against_real_forward(model, torch.randint(0, 256, (1, 32), device=device))

    max_round = args.max_round if args.max_round is not None else model.config.n_layer
    n_classes = len(LOCATIONS)
    results_by_hops = {}
    for n_hops in args.n_hops:
        print(f"\n=== n_hops={n_hops} ===", flush=True)
        train_examples = [generate_example(rng, n_hops) for _ in range(args.n_train)]
        test_examples = [generate_example(rng, n_hops) for _ in range(args.n_test)]

        def collect(examples):
            per_round_all = [[] for _ in range(max_round)]
            labels = []
            for text, answer_idx in examples:
                idx = torch.tensor([list(text.encode("utf-8"))], dtype=torch.long, device=device)
                per_round = collect_all_round_last_token(model, idx, n_rounds=max_round)
                for r, z in enumerate(per_round):
                    per_round_all[r].append(z)
                labels.append(answer_idx)
            per_round_tensors = [torch.cat(rs, dim=0) for rs in per_round_all]
            return per_round_tensors, torch.tensor(labels, dtype=torch.long)

        train_z_by_round, train_y = collect(train_examples)
        test_z_by_round, test_y = collect(test_examples)

        round_accs = []
        for r in range(max_round):
            acc = train_and_eval_probe(train_z_by_round[r], train_y, test_z_by_round[r], test_y,
                                        n_classes, args.probe_epochs, args.probe_lr)
            round_accs.append(acc)
            beyond = " (BEYOND trained depth)" if r + 1 > model.config.n_layer else ""
            print(f"  round {r+1}/{max_round}: probe test accuracy = {acc:.3f} "
                  f"(chance = {1/n_classes:.3f}){beyond}", flush=True)
        results_by_hops[n_hops] = round_accs

    report = {
        "checkpoint": str(args.checkpoint),
        "n_layer": model.config.n_layer,
        "max_round": max_round,
        "n_classes": n_classes,
        "chance_accuracy": 1 / n_classes,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "results_by_hops": results_by_hops,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
