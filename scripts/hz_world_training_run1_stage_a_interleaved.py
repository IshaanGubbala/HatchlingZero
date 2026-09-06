#!/usr/bin/env python3
"""Hatchling World Training Run 1, Stage A -- REHEARSAL variant (plans/
Hatchling world.md section 0.6's real follow-up to the sequential run's
catastrophic-forgetting finding). Same model, same stages, same TOTAL
per-stage step budgets, same retention-eval logic/seeds as
`hz_world_training_run1_stage_a.py` (imported, not duplicated) -- the
ONLY difference is scheduling: once a stage is introduced, it is never
fully abandoned. Each "phase" (named after the newest stage being
introduced) spends its step budget sampling UNIFORMLY at random among
every stage introduced so far (including the new one), one real
train_step per sampled stage per step -- standard task-interleaving
rehearsal, not a new mechanism.

Real question this answers: does rehearsal recover the sequential run's
catastrophic forgetting (L1 1.000->0.253, L2 sel_acc 1.000->0.287 below
its own chance floor, L3 unseen 0.68->0.00), at the same total step
budget?
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hz_nursery_train as nt  # noqa: E402
import hz_world_training_run1_stage_a as run1  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.tokenizer import NurseryTokenizer, NOVEL_LABELS  # noqa: E402


def make_stage_step_fns(model, opt, tok, args, rngs):
    """One zero-arg callable per stage, each closed over its OWN
    persistent rng (created once at the top of main, not per-phase) --
    sampling a stage mid-run always continues that stage's own episode
    sequence, it never restarts."""
    return {
        "L0": lambda: nt.l0_train_step(model, opt, tok, rngs["L0"], args.l0_batch_size),
        "L1": lambda: nt.l1_train_step(model, opt, tok, rngs["L1"], args.l1_n_objects),
        "L2": lambda: nt.l2_train_step(model, opt, tok, rngs["L2"], args.l2_n_objects),
        "L3": lambda: nt.l3_train_step(model, opt, tok, rngs["L3"], args.l3_n_objects),
        "L4-logic": lambda: nt.l4_logic_train_step(model, opt, tok, rngs["L4-logic"], args.l4_n_objects),
        "L4-counting": lambda: nt.l4_counting_train_step(model, opt, tok, rngs["L4-counting"], args.l4_n_objects),
        "L5": lambda: nt.l5_train_step(model, opt, tok, rngs["L5"], args.l5_n_objects),
        "L6": lambda: nt.l6_train_step(model, opt, tok, rngs["L6"], args.l6_n_sentences),
    }


PHASE_BUDGET_ARG = {
    "L0": "l0_steps", "L1": "l1_steps", "L2": "l2_steps", "L3": "l3_steps",
    "L4-logic": "l4_logic_steps", "L4-counting": "l4_counting_steps",
    "L5": "l5_steps", "L6": "l6_steps",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l0-steps", type=int, default=2000)
    parser.add_argument("--l0-batch-size", type=int, default=16)
    parser.add_argument("--l1-steps", type=int, default=1500)
    parser.add_argument("--l1-n-objects", type=int, default=4)
    parser.add_argument("--l2-steps", type=int, default=1500)
    parser.add_argument("--l2-n-objects", type=int, default=4)
    parser.add_argument("--l3-steps", type=int, default=2000)
    parser.add_argument("--l3-n-objects", type=int, default=4)
    parser.add_argument("--l4-logic-steps", type=int, default=2000)
    parser.add_argument("--l4-counting-steps", type=int, default=2000)
    parser.add_argument("--l4-n-objects", type=int, default=4)
    parser.add_argument("--l5-steps", type=int, default=2000)
    parser.add_argument("--l5-n-objects", type=int, default=4)
    parser.add_argument("--l6-steps", type=int, default=2000)
    parser.add_argument("--l6-n-sentences", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=150)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/local/hz_world_run1_stage_a_interleaved"))
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_world_run1_stage_a_interleaved_retention.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[stage-a-interleaved] PERSISTENT model created once: vocab_size={tok.vocab_size} n_params={n_params}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    stage_order = ["L0", "L1", "L2", "L3", "L4-logic", "L4-counting", "L5", "L6"]
    # Persistent per-stage rngs -- created ONCE, matching the sequential
    # script's own seed offsets exactly, so any given stage's episode
    # sequence is directly comparable in kind (not seed) to the
    # sequential run.
    rngs = {
        "L0": random.Random(args.seed + 1), "L1": random.Random(args.seed + 2),
        "L2": random.Random(args.seed + 3), "L3": random.Random(args.seed + 4),
        "L4-logic": random.Random(args.seed + 5), "L4-counting": random.Random(args.seed + 6),
        "L5": random.Random(args.seed + 7), "L6": random.Random(args.seed + 8),
    }
    schedule_rng = random.Random(args.seed + 999)
    step_fns = make_stage_step_fns(model, opt, tok, args, rngs)
    retention_fns = run1.make_retention_fns(tok, args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    retention_matrix = []
    introduced = []
    per_stage_calls = {s: 0 for s in stage_order}
    t_start = time.time()
    for new_stage in stage_order:
        introduced.append(new_stage)
        phase_steps = getattr(args, PHASE_BUDGET_ARG[new_stage])
        t0 = time.time()
        print(f"\n[stage-a-interleaved] ===== phase introducing {new_stage}: {phase_steps} steps, "
              f"sampled uniformly among {introduced} =====", flush=True)
        for step in range(phase_steps):
            chosen = schedule_rng.choice(introduced)
            step_fns[chosen]()
            per_stage_calls[chosen] += 1
            if (step + 1) % args.log_every == 0:
                print(f"[stage-a-interleaved][{new_stage} phase] step={step+1}/{phase_steps} "
                      f"last_sampled={chosen} calls_so_far={per_stage_calls}", flush=True)
        print(f"[stage-a-interleaved] {new_stage} phase done in {time.time()-t0:.0f}s. "
              f"Retention check on all {len(introduced)} introduced stages...", flush=True)

        scores = {s: retention_fns[s](model) for s in introduced}
        retention_matrix.append({"after_stage": new_stage, "scores": scores})
        print(f"[stage-a-interleaved] retention after {new_stage}: " +
              " | ".join(f"{s}={scores[s]}" for s in introduced), flush=True)

        ckpt_path = args.checkpoint_dir / f"after_{new_stage.replace('-', '_')}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[stage-a-interleaved] checkpoint saved: {ckpt_path}", flush=True)

    total_time = time.time() - t_start
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump({"stage_order": stage_order, "retention_matrix": retention_matrix,
                    "n_params": n_params, "total_seconds": total_time,
                    "per_stage_calls": per_stage_calls}, f, indent=2)
    print(f"\n[stage-a-interleaved] DONE in {total_time:.0f}s. Wrote {args.results_file}", flush=True)
    print(f"[stage-a-interleaved] total train calls per stage: {per_stage_calls}", flush=True)


if __name__ == "__main__":
    main()
