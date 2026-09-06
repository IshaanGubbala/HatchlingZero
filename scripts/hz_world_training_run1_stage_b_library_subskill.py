#!/usr/bin/env python3
"""Hatchling World Training Run 1 -- PER-SUBSKILL mastery scheduler,
user-directed fix to the exact regression found in `hz_world_training_
run1_stage_b_library.py`: a single scalar mastery per stage let
L4-logic's unseen-combo generalization decay 96.7% -> 54.7% (and L3's
100% -> 83.3%) because both stages' tracked mastery signal only ever
reflected SEEN-pair training accuracy (`l3_train_step`/`l4_logic_
train_step` only ever train on `split="train"` by design -- there is no
training-time signal for the unseen subskill at all).

Real fix: track mastery PER SUBSKILL, not one scalar per stage.
- Corpus, L0, L1, L5, L6, Library: single subskill ("main"), updated
  from each train_step's own returned accuracy (unchanged mechanism).
- L2: two subskills ("sel", "cons"), BOTH updated from train_step's own
  return every call (l2_train_step already trains and scores both
  heads every step -- no extra eval needed).
- L3, L4-logic: two subskills ("seen", "unseen"). "seen" updates from
  train_step's own return (trained every call). "unseen" has NO
  training-time signal at all (the training loop never samples
  split="test") -- refreshed instead via a small periodic held-out eval
  (`--subskill-eval-every` steps, `--subskill-eval-episodes` real
  episodes each time, real held-out data, not synthetic).

Stage priority: `need(stage) = max_j(1 - mastery_j)` over that stage's
own tracked subskills (a stage is only as "mastered" as its WORST
subskill) -- exactly the user's own specified formula, replacing the
single-scalar-per-stage priority from the previous scheduler.

Loads C_9 (NOT C_10 -- the same starting checkpoint as the previous
Library run, per the user's own instruction, for a clean, controlled
A/B comparison changing ONLY the scheduler) and reruns the exact same
2,000-step Library-introduction phase.

Pre-committed success criteria (stated BEFORE running, per this
project's own decisive-evaluation discipline): Library >= 0.95, L3
unseen >= 0.90, L4-logic unseen >= 0.90, while L1/L2 stay essentially
intact (>= 0.90). If met, promote this scheduler as the production
Hatchling World scheduler; if not, report the real shortfall honestly.
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
from hz_world_training_run1_stage_a_corpus import CorpusPool, corpus_train_step, corpus_eval  # noqa: E402
from hz_world_training_run1_stage_a_adaptive import weighted_choice, MASTERY_FLOOR  # noqa: E402
from hz_world_training_run1_stage_b_library import library_train_step, library_eval  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.language.tokenizer import NOVEL_LABELS  # noqa: E402

TEST_SEED_OFFSET = nt.TEST_SEED_OFFSET
EMA_DECAY = 0.95

# Which stages have a subskill with NO training-time signal, needing a
# periodic real held-out eval refresh instead: {stage: subskill_name}.
EVAL_ONLY_SUBSKILLS = {"L3": "unseen", "L4-logic": "unseen"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                         default=Path("results/local/hz_world_run1_stage_a_adaptive/after_L6.pt"))
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--corpus-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--corpus-val-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--corpus-max-lines", type=int, default=20000)
    parser.add_argument("--corpus-val-max-lines", type=int, default=2000)
    parser.add_argument("--corpus-eval-windows", type=int, default=100)
    parser.add_argument("--library-n-facts", type=int, default=20)
    parser.add_argument("--library-steps", type=int, default=2000)
    parser.add_argument("--l0-batch-size", type=int, default=16)
    parser.add_argument("--l1-n-objects", type=int, default=4)
    parser.add_argument("--l2-n-objects", type=int, default=4)
    parser.add_argument("--l3-n-objects", type=int, default=4)
    parser.add_argument("--l4-n-objects", type=int, default=4)
    parser.add_argument("--l5-n-objects", type=int, default=4)
    parser.add_argument("--l6-n-sentences", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=150)
    parser.add_argument("--subskill-eval-every", type=int, default=100,
                         help="How often (in scheduler steps) to refresh eval-only subskills (L3/L4-logic unseen).")
    parser.add_argument("--subskill-eval-episodes", type=int, default=20,
                         help="Real held-out episodes per refresh -- small and cheap, called often.")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/local/hz_world_run1_stage_b_library_subskill"))
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_world_run1_stage_b_library_subskill_retention.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    print(f"[stage-b-subskill] loading checkpoint (C_9, same starting point as the previous Library run "
          f"for a clean A/B): {args.checkpoint}", flush=True)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[stage-b-subskill] CONTINUING the persistent trainee: n_params={n_params}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    corpus_pool = CorpusPool(args.corpus_data, args.corpus_max_lines)
    corpus_val_pool = CorpusPool(args.corpus_val_data, args.corpus_val_max_lines)
    print(f"[stage-b-subskill] loaded {len(corpus_pool.windows)} corpus train windows", flush=True)

    old_stages = ["Corpus", "L0", "L1", "L2", "L3", "L4-logic", "L4-counting", "L5", "L6"]
    all_stages = old_stages + ["Library"]
    rngs = {
        "Corpus": random.Random(args.seed), "L0": random.Random(args.seed + 1), "L1": random.Random(args.seed + 2),
        "L2": random.Random(args.seed + 3), "L3": random.Random(args.seed + 4),
        "L4-logic": random.Random(args.seed + 5), "L4-counting": random.Random(args.seed + 6),
        "L5": random.Random(args.seed + 7), "L6": random.Random(args.seed + 8),
        "Library": random.Random(args.seed + 10),
    }
    # Separate, persistent (never-reset) rngs for the eval-only subskill
    # refreshes -- advances across the whole run so repeated small
    # refreshes sample different held-out episodes over time, not the
    # same handful repeatedly.
    subskill_eval_rngs = {"L3": random.Random(args.seed + 4 + TEST_SEED_OFFSET),
                           "L4-logic": random.Random(args.seed + 5 + TEST_SEED_OFFSET)}
    corpus_eval_rng = random.Random(args.seed + TEST_SEED_OFFSET)
    library_eval_rng = random.Random(args.seed + 10 + TEST_SEED_OFFSET)
    schedule_rng = random.Random(args.seed + 999)

    step_fns = {
        "Corpus": lambda: corpus_train_step(model, opt, corpus_pool, rngs["Corpus"], tok.vocab_size),
        "L0": lambda: nt.l0_train_step(model, opt, tok, rngs["L0"], args.l0_batch_size),
        "L1": lambda: nt.l1_train_step(model, opt, tok, rngs["L1"], args.l1_n_objects),
        "L2": lambda: nt.l2_train_step(model, opt, tok, rngs["L2"], args.l2_n_objects),
        "L3": lambda: nt.l3_train_step(model, opt, tok, rngs["L3"], args.l3_n_objects),
        "L4-logic": lambda: nt.l4_logic_train_step(model, opt, tok, rngs["L4-logic"], args.l4_n_objects),
        "L4-counting": lambda: nt.l4_counting_train_step(model, opt, tok, rngs["L4-counting"], args.l4_n_objects),
        "L5": lambda: nt.l5_train_step(model, opt, tok, rngs["L5"], args.l5_n_objects),
        "L6": lambda: nt.l6_train_step(model, opt, tok, rngs["L6"], args.l6_n_sentences),
        "Library": lambda: library_train_step(model, opt, tok, rngs["Library"], args.library_n_facts),
    }
    # Per-stage subskill layout and how each subskill's mastery updates.
    SUBSKILLS = {
        "Corpus": ["main"], "L0": ["main"], "L1": ["main"], "L5": ["main"], "L6": ["main"], "Library": ["main"],
        "L2": ["sel", "cons"], "L3": ["seen", "unseen"], "L4-logic": ["seen", "unseen"], "L4-counting": ["main"],
    }

    def refresh_eval_only(stage: str) -> float:
        if stage == "L3":
            return nt.l3_eval(model, tok, subskill_eval_rngs["L3"], args.l3_n_objects,
                               args.subskill_eval_episodes, split="test")
        if stage == "L4-logic":
            return nt.l4_logic_eval(model, tok, subskill_eval_rngs["L4-logic"], args.l4_n_objects,
                                     args.subskill_eval_episodes, split="test")
        raise ValueError(stage)

    retention_fns = run1.make_retention_fns(tok, args)
    retention_fns["Corpus"] = lambda m: {"held_out_next_byte_acc":
        corpus_eval(m, corpus_val_pool, corpus_eval_rng, tok.vocab_size, args.corpus_eval_windows)}
    retention_fns["Library"] = lambda m: {"held_out_acc":
        library_eval(m, tok, library_eval_rng, args.library_n_facts, args.eval_episodes)}
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n[stage-b-subskill] ===== BEFORE Library: retention on the inherited lineage (C_9) =====", flush=True)
    before_scores = {s: retention_fns[s](model) for s in old_stages}
    for s in old_stages:
        print(f"[stage-b-subskill] before Library, {s}={before_scores[s]}", flush=True)

    # mastery[stage][subskill] -- initialized from the REAL C_9 retention
    # scores just measured above, not assumed. Getting this wrong would
    # silently invert the whole point of per-subskill tracking (e.g.
    # Corpus's real ~0.40 mastery must start LOW, not be assumed
    # near-mastered like L1).
    def initial_mastery(stage: str) -> dict:
        s = before_scores[stage]
        if stage == "Corpus":
            return {"main": s["held_out_next_byte_acc"]}
        if stage == "L2":
            return {"sel": s["sel_acc"], "cons": s["cons_acc"]}
        if stage in ("L3", "L4-logic"):
            return {"seen": s["seen_combo_acc"], "unseen": s["unseen_combo_acc"]}
        acc_key = "next_token_acc" if stage == "L0" else "held_out_acc"
        return {"main": s[acc_key]}

    mastery = {s: initial_mastery(s) for s in old_stages}
    mastery["Library"] = {"main": 0.0}  # brand new, no signal yet -> max priority
    print(f"[stage-b-subskill] initial per-subskill mastery (from real C_9 scores): {mastery}", flush=True)

    print(f"\n[stage-b-subskill] ===== Library phase: {args.library_steps} steps, PER-SUBSKILL "
          f"mastery-weighted among all {len(all_stages)} stages (n_facts={args.library_n_facts}) =====", flush=True)
    per_stage_calls = {s: 0 for s in all_stages}
    t0 = time.time()
    for step in range(args.library_steps):
        # need(stage) = max_j(1 - m_j) = 1 - min_j(m_j) -- driven by the
        # WORST subskill, not the best (that inversion is exactly the
        # bug this script exists to fix: L4-logic's seen=1.0 must not
        # mask its unseen=0.55).
        needs = [max(1.0 - min(mastery[s].values()), MASTERY_FLOOR) for s in all_stages]
        chosen = weighted_choice(schedule_rng, all_stages, needs)
        result = step_fns[chosen]()
        if chosen == "L2":
            _, sel_acc, cons_acc = result
            mastery["L2"]["sel"] = EMA_DECAY * mastery["L2"]["sel"] + (1 - EMA_DECAY) * sel_acc
            mastery["L2"]["cons"] = EMA_DECAY * mastery["L2"]["cons"] + (1 - EMA_DECAY) * cons_acc
        elif chosen in ("L3", "L4-logic"):
            _, acc = result  # this is the SEEN-pair training accuracy
            mastery[chosen]["seen"] = EMA_DECAY * mastery[chosen]["seen"] + (1 - EMA_DECAY) * acc
        else:
            _, acc = result
            mastery[chosen]["main"] = EMA_DECAY * mastery[chosen]["main"] + (1 - EMA_DECAY) * acc
        per_stage_calls[chosen] += 1

        if (step + 1) % args.subskill_eval_every == 0:
            for stage, sub in EVAL_ONLY_SUBSKILLS.items():
                fresh = refresh_eval_only(stage)
                mastery[stage][sub] = EMA_DECAY * mastery[stage][sub] + (1 - EMA_DECAY) * fresh

        if (step + 1) % args.log_every == 0:
            mastery_str = " ".join(f"{s}={mastery[s]}" for s in all_stages)
            print(f"[stage-b-subskill] step={step+1}/{args.library_steps} last_sampled={chosen} "
                  f"mastery=[{mastery_str}] calls_so_far={per_stage_calls}", flush=True)
    print(f"[stage-b-subskill] Library phase done in {time.time()-t0:.0f}s.", flush=True)

    print("\n[stage-b-subskill] ===== AFTER Library: retention on ALL 10 stages =====", flush=True)
    after_scores = {s: retention_fns[s](model) for s in all_stages}
    for s in all_stages:
        print(f"[stage-b-subskill] after Library, {s}={after_scores[s]}", flush=True)

    ckpt_path = args.checkpoint_dir / "after_Library.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[stage-b-subskill] checkpoint saved: {ckpt_path}", flush=True)

    # Pre-committed verdict, printed automatically -- not left to
    # eyeballing the table after the fact.
    library_acc = after_scores["Library"]["held_out_acc"]
    l3_unseen = after_scores["L3"]["unseen_combo_acc"]
    l4_unseen = after_scores["L4-logic"]["unseen_combo_acc"]
    l1_acc = after_scores["L1"]["held_out_acc"]
    l2_sel = after_scores["L2"]["sel_acc"]
    passed = (library_acc >= 0.95 and l3_unseen >= 0.90 and l4_unseen >= 0.90
              and l1_acc >= 0.90 and l2_sel >= 0.90)
    print(f"\n[stage-b-subskill] === PRE-COMMITTED VERDICT ===")
    print(f"Library={library_acc:.3f} (need >=0.95)  L3_unseen={l3_unseen:.3f} (need >=0.90)  "
          f"L4logic_unseen={l4_unseen:.3f} (need >=0.90)  L1={l1_acc:.3f} (need >=0.90)  "
          f"L2_sel={l2_sel:.3f} (need >=0.90)")
    print(f"[stage-b-subskill] {'PASS -- promote as production scheduler' if passed else 'FAIL -- report honestly, do not promote'}", flush=True)

    with open(args.results_file, "w") as f:
        json.dump({"before_library": before_scores, "after_library": after_scores,
                    "per_stage_calls": per_stage_calls, "final_mastery": mastery,
                    "n_params": n_params, "passed": passed}, f, indent=2)
    print(f"\n[stage-b-subskill] DONE. Wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
