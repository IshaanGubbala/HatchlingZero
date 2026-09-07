#!/usr/bin/env python3
"""Hatchling World Run 2 interference fork -- Arm B: PROTECTED Chat-only
continuation (plans/Hatchling world.md section 0.6), user-directed.

Context: Run 2's 12k->36k continuation (Arm A, `hz_world_training_run2_
continue.py`, already run, NOT rerun here) showed Knowledge improving
cleanly (undertraining, fixed by more compute) while Chat improved
~10x more slowly and non-monotonically on the same shared 12-channel
budget. The open question this fork answers directly: is Chat's slow
progress caused by gradient interference from the other 11 channels,
or is it a data/capacity/objective limit that would show up either
way?

Arm B starts from the EXACT SAME `results/local/hz_world_run2/
step_12000.pt` checkpoint as Arm A, but trains ONLY on Chat data --
zero updates on any other channel, a fully protected block, not merely
"Chat-heavy." Evaluated at Chat-update counts matching Arm A's actual
cumulative Chat calls at each of its four eval points (839, 1700, 2554,
3485 -- read directly from Arm A's own results file, not re-derived),
so the two arms differ in exactly one respect: interleaved-vs-
protected gradients, at matched Chat-update counts, from the same
starting theta.

At each of the 4 checkpoints, evaluates the SAME full 12-channel
retention (L0-L6 etc. are pure eval here, never trained in this arm --
this is what actually measures whether protecting Chat destroys the
rest) and real chat generation + diversity metrics on the SAME 6 fixed
held-out prompts Arm A used, for a real, directly comparable table.

Real, disclosed limitation shared with Arm A: optimizer state was not
saved by Run 2, so AdamW is reinitialized fresh over the 12k weights
(same as Arm A -- keeps the two arms comparable to each other, even if
neither is a perfect momentum-preserving continuation).
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

import hz_world_training_run1_stage_a as run1  # noqa: E402
from hz_world_training_run1_stage_b_library import library_eval  # noqa: E402
from hz_world_training_run1_stage_b_knowledge import knowledge_eval_condition  # noqa: E402
from hz_chat_micro_sft_train import train_step as chat_train_step, eval_items as chat_eval_items, sample_generations  # noqa: E402
from hz_world_training_run2 import corpus_eval_loss  # noqa: E402
from hz_world_training_run2_continue import chat_diversity_metrics  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.language.tokenizer import NOVEL_LABELS  # noqa: E402
from hatchling_world.knowledge.facts import TRAIN_FACTS  # noqa: E402
from hatchling_world.knowledge.chat_data import build_chat_split  # noqa: E402
from hatchling_world.knowledge.squad_corpus import build_squad_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("results/local/hz_world_run2/step_12000.pt"))
    parser.add_argument("--arm-a-results", type=Path, default=Path("results/local/hz_world_run2_continuation.json"))
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--workspace-slots", type=int, default=64)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l1-n-objects", type=int, default=4)
    parser.add_argument("--l2-n-objects", type=int, default=4)
    parser.add_argument("--l3-n-objects", type=int, default=4)
    parser.add_argument("--l4-n-objects", type=int, default=4)
    parser.add_argument("--l5-n-objects", type=int, default=4)
    parser.add_argument("--l6-n-sentences", type=int, default=3)
    parser.add_argument("--library-n-facts", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--subskill-eval-episodes", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/local/hz_world_run2_fork_chat_only"))
    parser.add_argument("--results-file", type=Path,
                         default=Path("results/local/hz_world_run2_fork_chat_only.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    print(f"[fork-B] loading base checkpoint: {args.base_checkpoint}", flush=True)
    model.load_state_dict(torch.load(args.base_checkpoint, map_location="cpu"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[fork-B] ARM B -- PROTECTED CHAT-ONLY continuation: n_params={n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    with open(args.arm_a_results) as f:
        arm_a = json.load(f)
    chat_targets = [ep["calls_this_phase"]["Chat"] for ep in arm_a["eval_points"]]
    print(f"[fork-B] matched Chat-update targets from Arm A: {chat_targets}", flush=True)

    squad_train, squad_held, squad_paraphrase = build_squad_split(seed=args.seed)
    squad_held_contexts = sorted(set(item["context"] for item in squad_held))
    chat_train, chat_held = build_chat_split(seed=args.seed, max_train=300, max_held_out=50)
    print(f"[fork-B] real data (same as Run 2, same seed): {len(chat_train)} chat train, "
          f"{len(chat_held)} chat held-out", flush=True)

    chat_rng = random.Random(args.seed + 52)
    corpus_eval_rng = random.Random(args.seed + run1.TEST_SEED_OFFSET)
    library_eval_rng = random.Random(args.seed + 60 + run1.TEST_SEED_OFFSET)

    retention_fns = run1.make_retention_fns(tok, args)
    retention_fns["Corpus"] = lambda m: {"held_out_next_byte_acc": 1.0 - min(1.0, corpus_eval_loss(m, tok, squad_held_contexts) / 6.0)}
    retention_fns["Library"] = lambda m: {"held_out_acc":
        library_eval(m, tok, library_eval_rng, args.library_n_facts, args.eval_episodes)}
    retention_fns["Knowledge"] = lambda m: {"mean_loss": knowledge_eval_condition(m, tok, TRAIN_FACTS)["mean_loss"]}
    retention_fns["Chat"] = lambda m: chat_eval_items(m, tok, chat_held)
    stage_order = ["Corpus", "L0", "L1", "L2", "L3", "L4-logic", "L4-counting", "L5", "L6",
                   "Library", "Knowledge", "Chat"]

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_points = []
    total_chat_steps = 0
    t0 = time.time()

    for target in chat_targets:
        while total_chat_steps < target:
            chat_train_step(model, opt, tok, chat_rng.choice(chat_train))
            total_chat_steps += 1
            if total_chat_steps % args.log_every == 0:
                elapsed = time.time() - t0
                print(f"[fork-B] chat_calls={total_chat_steps}/{target} ({elapsed:.0f}s, "
                      f"{total_chat_steps/elapsed:.2f} steps/sec)", flush=True)

        ckpt_path = args.checkpoint_dir / f"chat_calls_{total_chat_steps}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[fork-B] checkpoint saved: {ckpt_path}", flush=True)

        print(f"\n[fork-B] ===== RETENTION @ chat_calls={total_chat_steps} "
              f"(all non-Chat channels are PURE EVAL -- never trained in this arm) =====", flush=True)
        final_scores = {}
        for s in stage_order:
            try:
                final_scores[s] = retention_fns[s](model)
                print(f"[fork-B] {s}: {final_scores[s]}", flush=True)
            except Exception as e:
                print(f"[fork-B] {s}: eval error {e}", flush=True)
                final_scores[s] = {"error": str(e)}

        print(f"\n[fork-B] ===== CHAT GENERATION @ chat_calls={total_chat_steps} "
              f"(same 6 fixed held-out prompts as Arm A) =====", flush=True)
        samples = sample_generations(model, tok, chat_held, n=6)
        for s in samples:
            print(f"[fork-B] Q: {s['instruction']}", flush=True)
            print(f"[fork-B]   generated: {s['generated']!r}", flush=True)
        diversity = chat_diversity_metrics(samples)
        print(f"[fork-B] chat diversity: {diversity}", flush=True)

        eval_points.append({
            "chat_calls": total_chat_steps,
            "final_scores": final_scores,
            "chat_diversity": diversity,
            "samples": samples,
            "elapsed_seconds": time.time() - t0,
        })
        with open(args.results_file, "w") as f:
            json.dump({"base_checkpoint": str(args.base_checkpoint), "n_params": n_params,
                        "matched_chat_targets": chat_targets, "eval_points": eval_points}, f, indent=2)
        print(f"[fork-B] wrote {args.results_file} ({len(eval_points)} eval points so far)\n", flush=True)

    total_time = time.time() - t0
    print(f"\n[fork-B] fork done: {total_chat_steps} chat-only steps in {total_time:.0f}s", flush=True)
    print(f"[fork-B] DONE.", flush=True)


if __name__ == "__main__":
    main()
