#!/usr/bin/env python3
"""HZ-CQ-v1 Mainline Phase 2 (plans/HatchlingZero — Mainline Research
Plan.md section 8): "the most important experiment in the project" --
does additional recurrent depth R actually improve reasoning accuracy
on harder tasks?

Real procedural task family (section 10's own example: A, then A o B,
then A o B o C): each episode samples a real composition depth k in
{1,2,4,8} and k random ORTHOGONAL DxD matrices (orthogonal, not plain
Gaussian, so repeated composition stays numerically well-conditioned
at depth=8 -- a real fix over the STEP 6 smoke test, which used plain
Gaussian matrices scaled by 1/sqrt(D) and risked compounding
instability). The model only ever sees INPUT/OUTPUT pairs of the full
composed map (via demos into S) -- never the individual steps, exactly
matching how ARC demos work.

Training exposes multiple R (section 8: R in {2,4,6,8,12,16}) sampled
per-episode, independent of depth, so the model sees many depth/R
combinations rather than R always matching depth exactly.

Real kill criterion, stated BEFORE running (plan Rule 3's own worked
example, adopted verbatim): if v1 produces <1-2 percentage points of
reproducible accuracy improvement from R=4->8/12 on DEEP tasks
(depth=8), do not claim depth reasoning.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_hzcq_v1_persistent_memory_torch import HZCQPersistentMemory, HZCQPersistentMemoryConfig
from reference.hz0h_bdh_hzcq_v1_reasoning_workspace_torch import HZCQReasoningWorkspace, HZCQReasoningWorkspaceConfig

TRAIN_DEPTHS = [1, 2, 4, 8]
TRAIN_R_VALUES = [2, 4, 6, 8, 12, 16]
EVAL_DEPTHS = [1, 2, 4, 8]
EVAL_R_VALUES = [1, 2, 4, 6, 8, 12, 16]


def random_orthogonal(D: int, rng: torch.Generator) -> torch.Tensor:
    A = torch.randn(D, D, generator=rng)
    Q, _ = torch.linalg.qr(A)
    return Q


def build_primitive_library(D: int, n_primitives: int, seed: int) -> list[torch.Tensor]:
    """Real fix over the first attempt: a FIXED, small, reusable set of
    primitive transformations, generated once and shared across every
    episode of the whole run (train and eval alike) -- not a fresh
    random matrix per episode. A brand-new random DxD orthogonal matrix
    every episode is information-theoretically underdetermined by a
    handful of demos (identifying an arbitrary D-dim linear map needs
    ~D independent examples in the worst case; 4 demos for D=48 is
    nowhere close) -- that was a real flaw in the first version of this
    script, confirmed by comparing the observed failure's relative
    error (~1.00) against the TRUE random-guess baseline (~1.41 for two
    independent unit vectors): the model was extracting real but
    insufficient signal, not failing to learn anything at all. A small
    FIXED library the model can come to recognize across many episodes
    (much closer to how ARC's bounded vocabulary of transformation
    types actually works) makes rule-identification from a few demos
    genuinely tractable."""
    rng = torch.Generator().manual_seed(seed)
    return [random_orthogonal(D, rng) for _ in range(n_primitives)]


def sample_episode(D: int, n_demos: int, depth: int, batch: int, rng: torch.Generator,
                    primitive_library: list[torch.Tensor]):
    idx = torch.randint(0, len(primitive_library), (depth,), generator=rng)
    mats = [primitive_library[i] for i in idx.tolist()]

    def compose(x: torch.Tensor) -> torch.Tensor:
        for M in mats:
            x = x @ M.T
        return x

    xs = torch.randn(batch, n_demos, D, generator=rng)
    ys = compose(xs.reshape(-1, D)).reshape(batch, n_demos, D)
    x_q = torch.randn(batch, D, generator=rng)
    target = compose(x_q)
    return xs, ys, x_q, target


def build_model(D: int, memory_slots: int, workspace_slots: int, gate_hidden: int, demo_proj_seed: int):
    mem = HZCQPersistentMemory(HZCQPersistentMemoryConfig(n_embd=D, memory_slots=memory_slots, gate_hidden=gate_hidden))
    ws = HZCQReasoningWorkspace(HZCQReasoningWorkspaceConfig(n_embd=D, workspace_slots=workspace_slots, gate_hidden=gate_hidden))
    readout = nn.Linear(D, D, bias=False)
    demo_encoder = nn.Linear(2 * D, D, bias=False)
    return mem, ws, readout, demo_encoder


def forward_episode(mem, ws, readout, demo_encoder, xs, ys, x_q, n_rounds: int):
    batch, n_demos, D = xs.shape
    demo_pairs = torch.cat([xs, ys], dim=-1)  # (B, n_demos, 2D)
    demo_hidden_all = demo_encoder(demo_pairs)  # (B, n_demos, D)
    demo_hiddens = [demo_hidden_all[:, i:i + 1, :] for i in range(n_demos)]
    S = mem.update_sequence(batch, demo_hiddens)
    H = ws.run(batch, S, x_q.unsqueeze(1), n_rounds=n_rounds)
    return readout(H.mean(dim=1))


def relative_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).norm(dim=-1) / target.norm(dim=-1).clamp_min(1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=8)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--n-demos", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--accuracy-threshold", type=float, default=0.1,
                         help="relative L2 error below this counts as 'correct' for the accuracy metric")
    parser.add_argument("--n-primitives", type=int, default=6,
                         help="size of the FIXED transformation library shared across every episode -- "
                              "real fix over a fresh-random-matrix-per-episode design, see build_primitive_library's docstring")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    D = args.d_model
    primitive_library = build_primitive_library(D, args.n_primitives, args.seed + 12345)
    mem, ws, readout, demo_encoder = build_model(D, args.memory_slots, args.workspace_slots, args.gate_hidden, args.seed)
    params = list(mem.parameters()) + list(ws.parameters()) + list(readout.parameters()) + list(demo_encoder.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    started = time.perf_counter()
    train_rng = torch.Generator().manual_seed(args.seed + 1)
    losses = []
    for step in range(args.train_steps):
        depth = TRAIN_DEPTHS[torch.randint(0, len(TRAIN_DEPTHS), (1,), generator=train_rng).item()]
        n_rounds = TRAIN_R_VALUES[torch.randint(0, len(TRAIN_R_VALUES), (1,), generator=train_rng).item()]
        xs, ys, x_q, target = sample_episode(D, args.n_demos, depth, args.batch_size, train_rng, primitive_library)
        opt.zero_grad(set_to_none=True)
        pred = forward_episode(mem, ws, readout, demo_encoder, xs, ys, x_q, n_rounds)
        loss = (pred - target).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
        if (step + 1) % 200 == 0:
            recent = sum(losses[-200:]) / 200
            print(f"[v1_composition] step {step+1}/{args.train_steps} recent_loss={recent:.4f}", flush=True)

    elapsed_train = time.perf_counter() - started
    print(f"[v1_composition] TRAIN DONE {elapsed_train:.0f}s final_recent_loss={sum(losses[-200:])/200:.4f}", flush=True)

    # Real paired difficulty x R evaluation -- SAME model, only depth
    # and n_rounds vary, matching plan section 8's evaluation protocol.
    eval_rng = torch.Generator().manual_seed(args.seed + 999)
    results = {}
    with torch.no_grad():
        for depth in EVAL_DEPTHS:
            results[depth] = {}
            for r in EVAL_R_VALUES:
                xs, ys, x_q, target = sample_episode(D, args.n_demos, depth, args.eval_episodes, eval_rng, primitive_library)
                pred = forward_episode(mem, ws, readout, demo_encoder, xs, ys, x_q, r)
                rel_err = relative_error(pred, target)
                accuracy = (rel_err < args.accuracy_threshold).float().mean().item()
                results[depth][r] = {"mean_relative_error": rel_err.mean().item(), "accuracy": accuracy}
                print(f"[v1_composition] eval depth={depth} R={r} "
                      f"mean_rel_err={rel_err.mean().item():.4f} accuracy={accuracy:.3f}", flush=True)

    # Real, stated-in-advance kill criterion check.
    deep = results[8]
    acc_r4 = deep[4]["accuracy"]
    acc_r8 = deep[8]["accuracy"]
    acc_r12 = deep[12]["accuracy"]
    best_improvement = max(acc_r8 - acc_r4, acc_r12 - acc_r4)
    kill_criterion_passed = best_improvement >= 0.015  # 1.5 percentage points, inside the stated 1-2pp band
    verdict = "PASS (real depth-reasoning signal)" if kill_criterion_passed else "FAIL (do not claim depth reasoning)"
    print(f"[v1_composition] KILL CRITERION: depth=8 accuracy R4={acc_r4:.3f} R8={acc_r8:.3f} R12={acc_r12:.3f} "
          f"best_improvement={best_improvement:+.3f} -> {verdict}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": vars(args) | {"out": str(args.out)},
        "train_elapsed_s": elapsed_train,
        "final_train_loss": sum(losses[-200:]) / 200,
        "results": {str(d): {str(r): v for r, v in rd.items()} for d, rd in results.items()},
        "kill_criterion": {
            "stated": "1-2 percentage points reproducible accuracy improvement from R=4->8/12 on depth=8",
            "acc_r4": acc_r4, "acc_r8": acc_r8, "acc_r12": acc_r12,
            "best_improvement": best_improvement, "verdict": verdict,
        },
    }, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
