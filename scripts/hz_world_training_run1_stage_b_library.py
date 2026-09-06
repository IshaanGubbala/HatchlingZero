#!/usr/bin/env python3
"""Hatchling World Training Run 1, Stage B first slice: Library
retrieval becomes load-bearing in the persistent trainee (plans/
Hatchling world.md section 0.6: "Phase 8's READ(query) mechanism,
already validated, becomes load-bearing here, not a toy").

Real, first-of-its-kind step for this whole "Run 1" line: every
Stage A script so far (sequential, interleaved, +corpus, adaptive)
started a FRESH model each time -- they were testing different
SCHEDULING mechanisms, not literally continuing one lineage. This
script LOADS the adaptive scheduler's own final checkpoint
(`results/local/hz_world_run1_stage_a_adaptive/after_L6.pt`, C_9 in the
plan's own C_0 -> C_1 -> ... -> C_n notation) and continues training
the SAME weights -- the first real checkpoint-continuation in this
project, not a fresh reinitialization.

Library is added as a 10th adaptive-scheduler channel, reusing
`HZLanguageModel.stress_recall_forward` UNCHANGED (now fixed to use
whole-sentence ingestion, see the memory-cliff production fix) --
Library's own episode shape (one retrieved fact + one question, no
object grounding) is already exactly what stress_recall_forward
expects, so no new model code is needed at all. The previous 9 stages
are all already "introduced" from the start of this run (they are
being CONTINUED, not re-introduced), so retention on them is checked
before AND after Library's phase, isolating what adding Library alone
does to the fully-trained lineage.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import hz_nursery_train as nt  # noqa: E402
import hz_world_training_run1_stage_a as run1  # noqa: E402
from hz_world_training_run1_stage_a_corpus import CorpusPool, corpus_train_step, corpus_eval  # noqa: E402
from hz_world_training_run1_stage_a_adaptive import weighted_choice, EMA_DECAY, MASTERY_FLOOR  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.language.tokenizer import NOVEL_LABELS  # noqa: E402
from hatchling_world.library import generate_library_episode, library_read  # noqa: E402

TEST_SEED_OFFSET = nt.TEST_SEED_OFFSET


def library_tensors(tok, ep):
    retrieved_label = library_read(ep["fact_table"], ep["query_color"])
    teach_ids = torch.tensor([tok.encode(f"the {ep['query_color']} object is called {retrieved_label}")])
    question_ids = torch.tensor([tok.encode(ep["question"])])
    target = torch.tensor([NOVEL_LABELS.index(ep["answer_label"])])
    return teach_ids, question_ids, target


def library_train_step(model, opt, tok, rng, n_facts):
    ep = generate_library_episode(rng, n_facts=n_facts)
    teach_ids, question_ids, target = library_tensors(tok, ep)
    logits = model.stress_recall_forward([teach_ids], question_ids)
    loss = F.cross_entropy(logits, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    acc = (logits.argmax(-1) == target).float().item()
    return loss.item(), acc


def library_eval(model, tok, rng, n_facts, n_episodes):
    correct = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_library_episode(rng, n_facts=n_facts)
            teach_ids, question_ids, target = library_tensors(tok, ep)
            logits = model.stress_recall_forward([teach_ids], question_ids)
            correct += int((logits.argmax(-1) == target).item())
    return correct / n_episodes


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
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/local/hz_world_run1_stage_b_library"))
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_world_run1_stage_b_library_retention.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    print(f"[stage-b-library] loading checkpoint (C_9 in the plan's own notation): {args.checkpoint}", flush=True)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[stage-b-library] CONTINUING the persistent trainee (not reinitializing): n_params={n_params}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    corpus_pool = CorpusPool(args.corpus_data, args.corpus_max_lines)
    corpus_val_pool = CorpusPool(args.corpus_val_data, args.corpus_val_max_lines)
    print(f"[stage-b-library] loaded {len(corpus_pool.windows)} corpus train windows", flush=True)

    old_stages = ["Corpus", "L0", "L1", "L2", "L3", "L4-logic", "L4-counting", "L5", "L6"]
    all_stages = old_stages + ["Library"]
    rngs = {
        "Corpus": random.Random(args.seed), "L0": random.Random(args.seed + 1), "L1": random.Random(args.seed + 2),
        "L2": random.Random(args.seed + 3), "L3": random.Random(args.seed + 4),
        "L4-logic": random.Random(args.seed + 5), "L4-counting": random.Random(args.seed + 6),
        "L5": random.Random(args.seed + 7), "L6": random.Random(args.seed + 8),
        "Library": random.Random(args.seed + 10),
    }
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
    retention_fns = run1.make_retention_fns(tok, args)
    retention_fns["Corpus"] = lambda m: {"held_out_next_byte_acc":
        corpus_eval(m, corpus_val_pool, corpus_eval_rng, tok.vocab_size, args.corpus_eval_windows)}
    retention_fns["Library"] = lambda m: {"held_out_acc":
        library_eval(m, tok, library_eval_rng, args.library_n_facts, args.eval_episodes)}
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ema_acc = {s: 1.0 for s in old_stages}  # inherited lineage: assume mastered until proven otherwise
    ema_acc["Library"] = 0.0  # brand new, max priority

    print("\n[stage-b-library] ===== BEFORE Library: retention on the inherited lineage (C_9) =====", flush=True)
    before_scores = {s: retention_fns[s](model) for s in old_stages}
    for s in old_stages:
        print(f"[stage-b-library] before Library, {s}={before_scores[s]}", flush=True)

    print(f"\n[stage-b-library] ===== Library phase: {args.library_steps} steps, mastery-weighted "
          f"among all {len(all_stages)} stages (n_facts={args.library_n_facts}) =====", flush=True)
    per_stage_calls = {s: 0 for s in all_stages}
    ACC_INDEX = 1
    t0 = time.time()
    for step in range(args.library_steps):
        weights = [max(1.0 - ema_acc[s], MASTERY_FLOOR) for s in all_stages]
        chosen = weighted_choice(schedule_rng, all_stages, weights)
        result = step_fns[chosen]()
        acc = result[ACC_INDEX]  # index 1 is the tracked mastery metric for every stage (L2: sel_acc)
        ema_acc[chosen] = EMA_DECAY * ema_acc[chosen] + (1 - EMA_DECAY) * acc
        per_stage_calls[chosen] += 1
        if (step + 1) % args.log_every == 0:
            mastery_str = " ".join(f"{s}={ema_acc[s]:.2f}" for s in all_stages)
            print(f"[stage-b-library] step={step+1}/{args.library_steps} last_sampled={chosen} "
                  f"ema_mastery=[{mastery_str}] calls_so_far={per_stage_calls}", flush=True)
    print(f"[stage-b-library] Library phase done in {time.time()-t0:.0f}s.", flush=True)

    print("\n[stage-b-library] ===== AFTER Library: retention on ALL 10 stages =====", flush=True)
    after_scores = {s: retention_fns[s](model) for s in all_stages}
    for s in all_stages:
        print(f"[stage-b-library] after Library, {s}={after_scores[s]}", flush=True)

    ckpt_path = args.checkpoint_dir / "after_Library.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[stage-b-library] checkpoint saved (C_10): {ckpt_path}", flush=True)

    with open(args.results_file, "w") as f:
        json.dump({"before_library": before_scores, "after_library": after_scores,
                    "per_stage_calls": per_stage_calls, "n_params": n_params}, f, indent=2)
    print(f"\n[stage-b-library] DONE. Wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
