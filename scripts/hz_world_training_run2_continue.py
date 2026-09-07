#!/usr/bin/env python3
"""Hatchling World Training Run 2 continuation (plans/Hatchling world.md
section 0.6): continues the SAME 5,056,229-param model from
`results/local/hz_world_run2/step_12000.pt` to 36,000 cumulative steps,
no architecture changes, no new data, no scheduler redesign -- exactly
the user's precommitted next experiment to answer "is Run 2 simply
undertrained?"

Real, disclosed mechanics:
  - Model weights loaded from the 12k checkpoint (not a fresh init).
  - Optimizer state was NOT saved by Run 2 and is NOT recoverable --
    AdamW is reinitialized fresh over the pretrained weights. Same
    disclosed limitation as HZ-Chat-Micro v1's pretrained-base SFT.
  - Scheduler mastery is WARM-STARTED from Run 2's final `final_mastery`
    (results/local/hz_world_run2_retention.json), not reset to 0 --
    this is a continuation, not a new cold-start run.
  - Per-stage RNG streams are re-seeded identically to Run 2 (exact
    mid-stream RNG state was not saved) -- real, disclosed limitation:
    each channel's early continuation steps replay the same example
    order Run 2 itself saw from step 0, rather than continuing a fresh
    random sequence. Does not bias the experiment (no channel sees
    systematically different data), just reduces example diversity
    versus a true unbroken RNG stream.
  - Same fixed 6 held-out chat prompts as Run 2's final report (same
    seed, same `build_chat_split` call -> same deterministic order) --
    required for a real apples-to-apples generation comparison across
    checkpoints.

Evaluates full retention (all 12 channels) + real chat generation +
chat-generation diversity metrics at every 6,000-step boundary:
18k, 24k, 30k, 36k cumulative steps. Diversity metrics (per the user's
explicit request, since loss alone can hide a collapsed-generation
failure): unique-generation count across the fixed prompt set,
average longest repeated-token run, and average consecutive-token
repetition rate.
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
from hz_world_training_run1_stage_a_adaptive import weighted_choice, EMA_DECAY, MASTERY_FLOOR  # noqa: E402
from hz_world_training_run1_stage_b_library import library_train_step, library_eval  # noqa: E402
from hz_world_training_run1_stage_b_knowledge import knowledge_train_step, knowledge_eval_condition  # noqa: E402
from hz_chat_micro_sft_train import (  # noqa: E402
    train_step as chat_train_step, eval_items as chat_eval_items, sample_generations,
)
from hz_world_training_run2 import corpus_train_step, corpus_eval_loss, EVAL_ONLY_SUBSKILLS  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.language.tokenizer import NOVEL_LABELS  # noqa: E402
from hatchling_world.knowledge.facts import TRAIN_FACTS  # noqa: E402
from hatchling_world.knowledge.chat_data import build_chat_split  # noqa: E402
from hatchling_world.knowledge.squad_corpus import build_squad_split  # noqa: E402

TEST_SEED_OFFSET = nt.TEST_SEED_OFFSET


def chat_diversity_metrics(samples: list) -> dict:
    generated = [s["generated"] for s in samples]
    unique_count = len(set(generated))

    def longest_repeated_token_run(s: str) -> int:
        toks = s.split()
        if not toks:
            return 0
        max_run = cur = 1
        for i in range(1, len(toks)):
            cur = cur + 1 if toks[i] == toks[i - 1] else 1
            max_run = max(max_run, cur)
        return max_run

    def token_repeat_rate(s: str) -> float:
        toks = s.split()
        if len(toks) <= 1:
            return 0.0
        reps = sum(1 for i in range(1, len(toks)) if toks[i] == toks[i - 1])
        return reps / (len(toks) - 1)

    runs = [longest_repeated_token_run(g) for g in generated]
    rates = [token_repeat_rate(g) for g in generated]
    return {
        "n_samples": len(generated),
        "unique_generations": unique_count,
        "avg_longest_repeated_token_run": sum(runs) / len(runs),
        "avg_token_repetition_rate": sum(rates) / len(rates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("results/local/hz_world_run2/step_12000.pt"))
    parser.add_argument("--base-results", type=Path, default=Path("results/local/hz_world_run2_retention.json"))
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--workspace-slots", type=int, default=64)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps-per-eval-point", type=int, default=6000)
    parser.add_argument("--n-eval-points", type=int, default=4)  # 12k -> 18k/24k/30k/36k
    parser.add_argument("--l0-batch-size", type=int, default=16)
    parser.add_argument("--l1-n-objects", type=int, default=4)
    parser.add_argument("--l2-n-objects", type=int, default=4)
    parser.add_argument("--l3-n-objects", type=int, default=4)
    parser.add_argument("--l4-n-objects", type=int, default=4)
    parser.add_argument("--l5-n-objects", type=int, default=4)
    parser.add_argument("--l6-n-sentences", type=int, default=3)
    parser.add_argument("--library-n-facts", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--subskill-eval-every", type=int, default=200)
    parser.add_argument("--subskill-eval-episodes", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/local/hz_world_run2"))
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_world_run2_continuation.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    print(f"[run2-cont] loading base checkpoint: {args.base_checkpoint}", flush=True)
    model.load_state_dict(torch.load(args.base_checkpoint, map_location="cpu"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[run2-cont] CONTINUING Run 2 (NOT a fresh init): n_params={n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    with open(args.base_results) as f:
        base_results = json.load(f)
    base_step = base_results["total_steps"]
    mastery = base_results["final_mastery"]
    base_calls = base_results["per_stage_calls"]
    print(f"[run2-cont] warm-started mastery + calls from step {base_step} "
          f"(NOT reset to 0 -- true continuation)", flush=True)

    squad_train, squad_held, squad_paraphrase = build_squad_split(seed=args.seed)
    squad_train_contexts = sorted(set(item["context"] for item in squad_train))
    squad_held_contexts = sorted(set(item["context"] for item in squad_held))
    chat_train, chat_held = build_chat_split(seed=args.seed, max_train=300, max_held_out=50)
    print(f"[run2-cont] real data (same as Run 2, same seed): {len(squad_train_contexts)} corpus paragraphs, "
          f"{len(TRAIN_FACTS)} knowledge facts, {len(chat_train)} chat examples, "
          f"{len(squad_held_contexts)} held-out corpus paragraphs", flush=True)

    stage_order = ["Corpus", "L0", "L1", "L2", "L3", "L4-logic", "L4-counting", "L5", "L6",
                   "Library", "Knowledge", "Chat"]
    rngs = {s: random.Random(args.seed + i) for i, s in enumerate(stage_order)}
    corpus_rng = random.Random(args.seed + 50)
    knowledge_rng = random.Random(args.seed + 51)
    chat_rng = random.Random(args.seed + 52)
    subskill_eval_rngs = {"L3": random.Random(args.seed + 4 + TEST_SEED_OFFSET),
                           "L4-logic": random.Random(args.seed + 5 + TEST_SEED_OFFSET)}
    corpus_eval_rng = random.Random(args.seed + TEST_SEED_OFFSET)
    library_eval_rng = random.Random(args.seed + 60 + TEST_SEED_OFFSET)
    schedule_rng = random.Random(args.seed + 999)

    step_fns = {
        "Corpus": lambda: corpus_train_step(model, opt, tok, corpus_rng.choice(squad_train_contexts)),
        "L0": lambda: nt.l0_train_step(model, opt, tok, rngs["L0"], args.l0_batch_size),
        "L1": lambda: nt.l1_train_step(model, opt, tok, rngs["L1"], args.l1_n_objects),
        "L2": lambda: nt.l2_train_step(model, opt, tok, rngs["L2"], args.l2_n_objects),
        "L3": lambda: nt.l3_train_step(model, opt, tok, rngs["L3"], args.l3_n_objects),
        "L4-logic": lambda: nt.l4_logic_train_step(model, opt, tok, rngs["L4-logic"], args.l4_n_objects),
        "L4-counting": lambda: nt.l4_counting_train_step(model, opt, tok, rngs["L4-counting"], args.l4_n_objects),
        "L5": lambda: nt.l5_train_step(model, opt, tok, rngs["L5"], args.l5_n_objects),
        "L6": lambda: nt.l6_train_step(model, opt, tok, rngs["L6"], args.l6_n_sentences),
        "Library": lambda: library_train_step(model, opt, tok, rngs["Library"], args.library_n_facts),
        "Knowledge": lambda: knowledge_train_step(model, opt, tok, knowledge_rng),
        "Chat": lambda: chat_train_step(model, opt, tok, chat_rng.choice(chat_train)),
    }

    def refresh_eval_only(stage: str) -> float:
        if stage == "L3":
            return nt.l3_eval(model, tok, subskill_eval_rngs["L3"], args.l3_n_objects,
                               args.subskill_eval_episodes, split="test")
        return nt.l4_logic_eval(model, tok, subskill_eval_rngs["L4-logic"], args.l4_n_objects,
                                 args.subskill_eval_episodes, split="test")

    retention_fns = run1.make_retention_fns(tok, args)
    retention_fns["Corpus"] = lambda m: {"held_out_next_byte_acc": 1.0 - min(1.0, corpus_eval_loss(m, tok, squad_held_contexts) / 6.0)}
    retention_fns["Library"] = lambda m: {"held_out_acc":
        library_eval(m, tok, library_eval_rng, args.library_n_facts, args.eval_episodes)}
    retention_fns["Knowledge"] = lambda m: {"mean_loss": knowledge_eval_condition(m, tok, TRAIN_FACTS)["mean_loss"]}
    retention_fns["Chat"] = lambda m: chat_eval_items(m, tok, chat_held)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    per_stage_calls = {s: 0 for s in stage_order}  # continuation-only calls; cumulative = this + base_calls
    total_continuation_steps = args.steps_per_eval_point * args.n_eval_points

    eval_points = []
    t0 = time.time()
    for step in range(total_continuation_steps):
        needs = [max(1.0 - min(mastery[s].values()), MASTERY_FLOOR) for s in stage_order]
        chosen = weighted_choice(schedule_rng, stage_order, needs)
        result = step_fns[chosen]()
        acc = result[1]
        mastery[chosen]["main"] = EMA_DECAY * mastery[chosen]["main"] + (1 - EMA_DECAY) * acc
        per_stage_calls[chosen] += 1

        if (step + 1) % args.subskill_eval_every == 0:
            for stage in EVAL_ONLY_SUBSKILLS:
                mastery.setdefault(stage, {})[EVAL_ONLY_SUBSKILLS[stage]] = refresh_eval_only(stage)

        cumulative_step = base_step + step + 1
        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"[run2-cont] cumulative_step={cumulative_step} (+{step+1}/{total_continuation_steps} "
                  f"continuation, {elapsed:.0f}s, {(step+1)/elapsed:.2f} steps/sec) last={chosen} "
                  f"mastery=[{ {s: round(v['main'],3) for s,v in mastery.items()} }] "
                  f"calls_this_phase={per_stage_calls}", flush=True)

        if (step + 1) % args.steps_per_eval_point == 0:
            ckpt_path = args.checkpoint_dir / f"step_{cumulative_step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[run2-cont] checkpoint saved: {ckpt_path}", flush=True)

            print(f"\n[run2-cont] ===== RETENTION @ cumulative_step={cumulative_step} =====", flush=True)
            final_scores = {}
            for s in stage_order:
                try:
                    final_scores[s] = retention_fns[s](model)
                    print(f"[run2-cont] {s}: {final_scores[s]}", flush=True)
                except Exception as e:
                    print(f"[run2-cont] {s}: eval error {e}", flush=True)
                    final_scores[s] = {"error": str(e)}

            print(f"\n[run2-cont] ===== CHAT GENERATION @ cumulative_step={cumulative_step} "
                  f"(same 6 fixed held-out prompts every eval point) =====", flush=True)
            samples = sample_generations(model, tok, chat_held, n=6)
            for s in samples:
                print(f"[run2-cont] Q: {s['instruction']}", flush=True)
                print(f"[run2-cont]   generated: {s['generated']!r}", flush=True)
            diversity = chat_diversity_metrics(samples)
            print(f"[run2-cont] chat diversity: {diversity}", flush=True)

            cumulative_calls = {s: base_calls[s] + per_stage_calls[s] for s in stage_order}
            eval_points.append({
                "cumulative_step": cumulative_step,
                "final_scores": final_scores,
                "chat_diversity": diversity,
                "samples": samples,
                "calls_this_phase": dict(per_stage_calls),
                "calls_cumulative": cumulative_calls,
                "mastery": {s: dict(v) for s, v in mastery.items()},
                "elapsed_seconds": time.time() - t0,
            })
            with open(args.results_file, "w") as f:
                json.dump({"base_checkpoint": str(args.base_checkpoint), "base_step": base_step,
                            "n_params": n_params, "eval_points": eval_points}, f, indent=2)
            print(f"[run2-cont] wrote {args.results_file} ({len(eval_points)} eval points so far)\n", flush=True)

    total_time = time.time() - t0
    print(f"\n[run2-cont] continuation done: {total_continuation_steps} steps in {total_time:.0f}s "
          f"(cumulative_step={base_step + total_continuation_steps})", flush=True)
    print(f"[run2-cont] DONE.", flush=True)


if __name__ == "__main__":
    main()
