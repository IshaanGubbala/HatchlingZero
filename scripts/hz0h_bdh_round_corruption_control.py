#!/usr/bin/env python3
"""Real corruption control, proposed 2026-08-29 as the natural follow-up
to the R-scaling matrix result (plans/HatchlingZero_Internal_Computation_Phase_2026-08-29.md
section 8: accuracy peaks at R=2-4 and DECLINES by R=8, no
A_1<A_2<A_4<A_8 signature). That result alone can't distinguish "later
rounds do nothing" from "later rounds actively hurt" -- this control
can. Reuses the already-trained backbone checkpoint from
scripts/hz0h_bdh_variable_depth_answer_train.py (no new backbone
training).

Real, disclosed gap this script works around: that training script
strips `answer_head` before saving (temporary/discardable, same
convention as this project's other per-experiment heads) -- so a
FRESH answer_head is fit here via real gradient descent on the FROZEN
backbone (only the head's own parameters ever get gradient; the
backbone forward runs entirely under torch.no_grad(), so its weights
are provably untouched). This is not just a workaround -- it directly
matches the "measurement instrument, not training signal" probing
methodology the plan itself called out as the right way to test
"does the state contain information" separately from "did probe
gradients force it to."

For each corruption round r in 1..R and each corruption type, run the
SAME real forward computation but with round r's update either
SKIPPED (h_r := h_{r-1}, that round's computation is thrown away) or
ZEROED (h_r := 0, a harder stress test of recovery), every OTHER round
proceeding normally, then compare accuracy against the real,
uncorrupted baseline at the same R.

Real, disclosed interpretation guide:
- skipping/zeroing round r hurts a LOT -> that round's own computation
  is genuinely load-bearing.
- skipping/zeroing round r barely changes accuracy -> that round isn't
  doing useful work at that position.
- skipping/zeroing round r IMPROVES accuracy over the uncorrupted
  baseline -> that round is actively harmful, not merely useless --
  directly explains a R=8 < R=4 decline if it's concentrated in the
  later rounds.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_shortcut_resistant_chain_task import generate_chain_example
from scripts.hz0h_bdh_variable_depth_answer_train import _iteration, add_answer_head
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device


def load_model(checkpoint_path: Path, device) -> BDHVBSubspaceDecoder:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = BDHVBSubspaceDecoderConfig(**ckpt["config"])
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.float32)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def backbone_last_token(model: BDHVBSubspaceDecoder, idx: torch.Tensor, n_rounds: int,
                         corrupt_round: int | None, corrupt_type: str) -> torch.Tensor:
    """Always runs under torch.no_grad() -- this script never trains the
    backbone, only ever (optionally) an answer_head sitting on top of a
    detached representation. corrupt_round is 1-indexed."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh
    with torch.no_grad():
        x = model.embed(idx).unsqueeze(1)
        x = model.ln(x)
        for r in range(1, n_rounds + 1):
            if corrupt_round is not None and r == corrupt_round:
                if corrupt_type == "skip":
                    pass  # x unchanged -- this round's real update is thrown away
                elif corrupt_type == "zero":
                    x = torch.zeros_like(x)
                else:
                    raise ValueError(corrupt_type)
            else:
                x = _iteration(x, model, B, T, D, nh, N)
        last_token = x[:, :, -1, :].reshape(B, D)
    return last_token.detach()


def fit_fresh_answer_head(model: BDHVBSubspaceDecoder, n_rounds: int, n_classes: int,
                           probe_fit_steps: int, seed: int, device, log_every: int = 200) -> None:
    add_answer_head(model, n_classes=n_classes)
    optimizer = torch.optim.Adam([model.answer_head], lr=1e-2, weight_decay=1e-3)
    rng = random.Random(seed)
    for step in range(probe_fit_steps):
        n_hops = rng.choice([1, 2, 3, 4, 6, 8])
        text, correct_idx, _ = generate_chain_example(rng, n_hops)
        idx = torch.tensor([list(text.encode("utf-8"))], dtype=torch.long, device=device)
        label = torch.tensor([correct_idx], dtype=torch.long, device=device)
        last_token = backbone_last_token(model, idx, n_rounds, None, "skip")  # real, uncorrupted, backbone-frozen
        logits = last_token @ model.answer_head  # only this op has a live grad path
        loss = F.cross_entropy(logits, label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"[probe_fit] step {step+1}/{probe_fit_steps} loss={float(loss):.4f}", flush=True)


@torch.no_grad()
def eval_accuracy(model: BDHVBSubspaceDecoder, n_hops: int, n_rounds: int, corrupt_round: int | None,
                   corrupt_type: str, eval_n: int, seed: int, device) -> float:
    rng = random.Random(seed)
    correct = 0
    for _ in range(eval_n):
        text, correct_idx, _shortcut_idx = generate_chain_example(rng, n_hops)
        idx = torch.tensor([list(text.encode("utf-8"))], dtype=torch.long, device=device)
        last_token = backbone_last_token(model, idx, n_rounds, corrupt_round, corrupt_type)
        logits = last_token @ model.answer_head
        if int(logits.argmax(dim=-1).item()) == correct_idx:
            correct += 1
    return correct / eval_n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/local/hz0h_bdh_variable_depth_answer_checkpoint.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-rounds", type=int, default=8)
    parser.add_argument("--n-hops", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--eval-n", type=int, default=80)
    parser.add_argument("--probe-fit-steps", type=int, default=3000)
    parser.add_argument("--n-classes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    device = pick_device(args.device)
    model = load_model(args.checkpoint, device)

    print(f"=== fitting fresh answer head at n_rounds={args.n_rounds} (backbone frozen) ===", flush=True)
    fit_fresh_answer_head(model, args.n_rounds, args.n_classes, args.probe_fit_steps, args.seed, device)

    results = {}
    for n_hops in args.n_hops:
        print(f"\n=== hops={n_hops}, n_rounds={args.n_rounds} ===", flush=True)
        seed_offset = args.seed + 999 + n_hops * 100
        baseline = eval_accuracy(model, n_hops, args.n_rounds, None, "skip", args.eval_n, seed_offset, device)
        print(f"  baseline (uncorrupted): accuracy={baseline:.3f}", flush=True)
        hop_results = {"baseline": baseline, "skip": {}, "zero": {}}
        for corrupt_type in ["skip", "zero"]:
            for r in range(1, args.n_rounds + 1):
                acc = eval_accuracy(model, n_hops, args.n_rounds, r, corrupt_type, args.eval_n, seed_offset, device)
                delta = acc - baseline
                hop_results[corrupt_type][r] = {"accuracy": acc, "delta_vs_baseline": delta}
                print(f"  corrupt_type={corrupt_type} round={r}: accuracy={acc:.3f} delta={delta:+.3f}", flush=True)
        results[n_hops] = hop_results

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_rounds": args.n_rounds, "results": results}, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
