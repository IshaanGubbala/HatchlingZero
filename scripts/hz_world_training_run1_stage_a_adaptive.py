#!/usr/bin/env python3
"""Hatchling World Training Run 1, Stage A -- ADAPTIVE MASTERY-WEIGHTED
scheduler (plans/Hatchling world.md section 0.7, step 3 of the sequenced
plan). Real, user-identified problem with the interleaved-rehearsal
scheduler (`hz_world_training_run1_stage_a_corpus.py`): uniform sampling
(P_i = 1/N) wastes training capacity on already-mastered stages equally
with unsolved ones -- the direct mechanism behind the earlier L4-logic
dilution finding (0.40 -> 0.027 unseen-combo, simply because it got a
smaller uniform share once more stages were introduced).

Real, minimal fix, not the full "learning need" formula (mastery +
improvement rate + forgetting + difficulty + time-since-rehearsal) --
that is real future work. This implements the single, load-bearing
piece: track an online EMA of each introduced stage's own recent
training accuracy, and sample stages with probability proportional to
(1 - mastery), floored so a fully-mastered stage still gets occasional
rehearsal (preventing the exact kind of neglect-driven forgetting the
floor is designed to catch, e.g. L1 dropping to chance in the ORIGINAL
sequential run once training moved on).

Same model, same stages (Corpus + L0-L6), same total per-phase step
budgets, same ByteTokenizer, same retention-eval logic as
`hz_world_training_run1_stage_a_corpus.py` -- the ONLY change is HOW a
stage is picked each step within a phase.
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
from hz_world_training_run1_stage_a_corpus import CorpusPool, corpus_train_step, PHASE_BUDGET_ARG  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.language.tokenizer import NOVEL_LABELS  # noqa: E402

# Which element of each train_step's return tuple is "the" accuracy to
# track mastery on (L2 returns (loss, sel_acc, cons_acc) -- sel_acc is
# the harder, more informative sub-skill; everything else returns
# (loss, acc)).
ACC_INDEX = {"Corpus": 1, "L0": 1, "L1": 1, "L2": 1, "L3": 1,
             "L4-logic": 1, "L4-counting": 1, "L5": 1, "L6": 1}

EMA_DECAY = 0.95
MASTERY_FLOOR = 0.05  # a fully-mastered stage (acc=1.0) still gets this much relative weight


def make_stage_step_fns(model, opt, tok, args, rngs, corpus_pool):
    return {
        "Corpus": lambda: corpus_train_step(model, opt, corpus_pool, rngs["Corpus"], tok.vocab_size),
        "L0": lambda: nt.l0_train_step(model, opt, tok, rngs["L0"], args.l0_batch_size),
        "L1": lambda: nt.l1_train_step(model, opt, tok, rngs["L1"], args.l1_n_objects),
        "L2": lambda: nt.l2_train_step(model, opt, tok, rngs["L2"], args.l2_n_objects),
        "L3": lambda: nt.l3_train_step(model, opt, tok, rngs["L3"], args.l3_n_objects),
        "L4-logic": lambda: nt.l4_logic_train_step(model, opt, tok, rngs["L4-logic"], args.l4_n_objects),
        "L4-counting": lambda: nt.l4_counting_train_step(model, opt, tok, rngs["L4-counting"], args.l4_n_objects),
        "L5": lambda: nt.l5_train_step(model, opt, tok, rngs["L5"], args.l5_n_objects),
        "L6": lambda: nt.l6_train_step(model, opt, tok, rngs["L6"], args.l6_n_sentences),
    }


def weighted_choice(rng: random.Random, items: list[str], weights: list[float]) -> str:
    total = sum(weights)
    r = rng.random() * total
    upto = 0.0
    for item, w in zip(items, weights):
        upto += w
        if upto >= r:
            return item
    return items[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--corpus-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--corpus-val-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--corpus-max-lines", type=int, default=20000)
    parser.add_argument("--corpus-val-max-lines", type=int, default=2000)
    parser.add_argument("--corpus-steps", type=int, default=2000)
    parser.add_argument("--corpus-eval-windows", type=int, default=100)
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
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/local/hz_world_run1_stage_a_adaptive"))
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_world_run1_stage_a_adaptive_retention.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[stage-a-adaptive] PERSISTENT model (byte-level, vocab_size={tok.vocab_size}) created once: "
          f"n_params={n_params}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"[stage-a-adaptive] loading real corpus: {args.corpus_data} (max {args.corpus_max_lines} windows)...", flush=True)
    corpus_pool = CorpusPool(args.corpus_data, args.corpus_max_lines)
    corpus_val_pool = CorpusPool(args.corpus_val_data, args.corpus_val_max_lines)
    print(f"[stage-a-adaptive] loaded {len(corpus_pool.windows)} train windows, "
          f"{len(corpus_val_pool.windows)} val windows", flush=True)

    stage_order = ["Corpus", "L0", "L1", "L2", "L3", "L4-logic", "L4-counting", "L5", "L6"]
    rngs = {
        "Corpus": random.Random(args.seed), "L0": random.Random(args.seed + 1), "L1": random.Random(args.seed + 2),
        "L2": random.Random(args.seed + 3), "L3": random.Random(args.seed + 4),
        "L4-logic": random.Random(args.seed + 5), "L4-counting": random.Random(args.seed + 6),
        "L5": random.Random(args.seed + 7), "L6": random.Random(args.seed + 8),
    }
    corpus_eval_rng = random.Random(args.seed + nt.TEST_SEED_OFFSET)
    schedule_rng = random.Random(args.seed + 999)
    step_fns = make_stage_step_fns(model, opt, tok, args, rngs, corpus_pool)
    retention_fns = run1.make_retention_fns(tok, args)

    def corpus_retention(m):
        from hz_world_training_run1_stage_a_corpus import corpus_eval
        return {"held_out_next_byte_acc": corpus_eval(m, corpus_val_pool, corpus_eval_rng, tok.vocab_size, args.corpus_eval_windows)}
    retention_fns["Corpus"] = corpus_retention
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ema_acc = {s: 0.0 for s in stage_order}  # 0.0 == "unknown, assume unmastered" -> max initial priority
    retention_matrix = []
    introduced = []
    per_stage_calls = {s: 0 for s in stage_order}
    t_start = time.time()
    for new_stage in stage_order:
        introduced.append(new_stage)
        phase_steps = getattr(args, PHASE_BUDGET_ARG[new_stage])
        t0 = time.time()
        print(f"\n[stage-a-adaptive] ===== phase introducing {new_stage}: {phase_steps} steps, "
              f"mastery-weighted among {introduced} =====", flush=True)
        for step in range(phase_steps):
            weights = [max(1.0 - ema_acc[s], MASTERY_FLOOR) for s in introduced]
            chosen = weighted_choice(schedule_rng, introduced, weights)
            result = step_fns[chosen]()
            acc = result[ACC_INDEX[chosen]]
            ema_acc[chosen] = EMA_DECAY * ema_acc[chosen] + (1 - EMA_DECAY) * acc
            per_stage_calls[chosen] += 1
            if (step + 1) % args.log_every == 0:
                mastery_str = " ".join(f"{s}={ema_acc[s]:.2f}" for s in introduced)
                print(f"[stage-a-adaptive][{new_stage} phase] step={step+1}/{phase_steps} "
                      f"last_sampled={chosen} ema_mastery=[{mastery_str}] calls_so_far={per_stage_calls}", flush=True)
        print(f"[stage-a-adaptive] {new_stage} phase done in {time.time()-t0:.0f}s. "
              f"Retention check on all {len(introduced)} introduced stages...", flush=True)

        scores = {s: retention_fns[s](model) for s in introduced}
        retention_matrix.append({"after_stage": new_stage, "scores": scores,
                                   "ema_mastery_at_end": dict(ema_acc)})
        print(f"[stage-a-adaptive] retention after {new_stage}: " +
              " | ".join(f"{s}={scores[s]}" for s in introduced), flush=True)

        ckpt_path = args.checkpoint_dir / f"after_{new_stage.replace('-', '_')}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[stage-a-adaptive] checkpoint saved: {ckpt_path}", flush=True)

    total_time = time.time() - t_start
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump({"stage_order": stage_order, "retention_matrix": retention_matrix,
                    "n_params": n_params, "total_seconds": total_time,
                    "per_stage_calls": per_stage_calls}, f, indent=2)
    print(f"\n[stage-a-adaptive] DONE in {total_time:.0f}s. Wrote {args.results_file}", flush=True)
    print(f"[stage-a-adaptive] total train calls per stage: {per_stage_calls}", flush=True)


if __name__ == "__main__":
    main()
