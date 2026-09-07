#!/usr/bin/env python3
"""Hatchling World Run 2 fork -- Arm D: moderate Chat breadth with
MATCHED exposure per example (plans/Hatchling world.md section 0.6),
user-directed. Context: Arm C (full pool, 1,741 examples, same 24k-step
budget as Arm A) improved Chat loss over Arm A with no collateral
damage, but generation stayed fully degenerate -- and Arm C's own Chat
channel averaged only ~2.92 exposures/example versus Arm A's ~16.1, a
confound this fork exists to remove. If breadth was the real lever,
Arm C should have shown it despite low repetition; it didn't. Arm D
tests breadth WITHOUT sacrificing repetition: a moderate pool (300 old
+ 300 new = 600 real examples + 4 refusals = 604 total, verified
byte-identical prefix to Arm A's exact 304-example pool) trained long
enough that the FULL pool reaches roughly Arm A's own ~16 exposures/
example average.

Real, disclosed mechanics:
  - Starts from the IDENTICAL `step_12000.pt` checkpoint as Arm A/B/C.
  - The "old" 304 examples already carry ~4.6 average exposures baked
    into the 12k checkpoint (1,409 Chat calls / 304 examples, read
    from Run 2's own retention file, not re-derived). The "new" 300
    examples have zero prior exposure. A CATCH-UP sampler exclusively
    draws from the new pool until its own average call count (this
    phase) reaches that ~4.6 baseline, then switches to uniform
    sampling across all 604 -- exactly the mechanism the user
    described in plain language, implemented as a small stateful
    picker (`make_chat_picker`), not a scheduler change.
  - Reaching ~16 exposures/example on a 604-item pool needs ~9,664
    total Chat-channel calls; 1,409 are already baked in, so this
    phase needs ~8,255 more. Chat's natural share of the full-HW
    scheduler was empirically ~14.5% of total steps in Arm A and Arm C
    (not assumed -- read from their own logs), so this run uses a
    60,000-step continuation budget (~8,700 expected Chat calls) --
    real, disclosed, ~3.5-4x Arm A/C's 24k-step budget, ~3.5-4 hours
    real wall-clock at the same ~4.6-4.7 steps/sec this setup has
    consistently shown.
  - Same 12-channel interleaved scheduler as A/B/C -- Hatchling World
    is not being abandoned, only Chat's own train-pool composition and
    the total step budget change.
  - Same 30 fixed held-out generation prompts, same diversity/
    coherence/relevance heuristics as Arm C (the naive real-word-
    fraction heuristic is known-gameable, disclosed again here;
    manual reading of the raw samples remains the real check).
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
from hz_world_training_run2_continue import chat_diversity_metrics  # noqa: E402
from hz_world_run2_fork_large_chat import (  # noqa: E402
    load_wordlist, coherent_english_rate, semantic_relevance_heuristic,
)
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.language.tokenizer import NOVEL_LABELS  # noqa: E402
from hatchling_world.knowledge.facts import TRAIN_FACTS  # noqa: E402
from hatchling_world.knowledge.chat_data import build_chat_split  # noqa: E402
from hatchling_world.knowledge.squad_corpus import build_squad_split  # noqa: E402

TEST_SEED_OFFSET = nt.TEST_SEED_OFFSET


def make_chat_picker(old_pool: list, new_pool: list, old_baseline_exposure: float, rng: random.Random):
    combined = old_pool + new_pool
    state = {"phase": "catchup", "new_calls": 0, "old_calls": 0, "uniform_calls": 0}

    def pick() -> dict:
        if state["phase"] == "catchup":
            avg_new = state["new_calls"] / len(new_pool)
            if avg_new < old_baseline_exposure:
                state["new_calls"] += 1
                return rng.choice(new_pool)
            state["phase"] = "uniform"
            print(f"[fork-D] catch-up complete: new pool avg exposure {avg_new:.2f} "
                  f">= old baseline {old_baseline_exposure:.2f} -- switching to uniform sampling "
                  f"over all {len(combined)} examples", flush=True)
        state["uniform_calls"] += 1
        return rng.choice(combined)

    return pick, state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("results/local/hz_world_run2/step_12000.pt"))
    parser.add_argument("--base-results", type=Path, default=Path("results/local/hz_world_run2_retention.json"))
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--workspace-slots", type=int, default=64)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps-per-eval-point", type=int, default=15000)
    parser.add_argument("--n-eval-points", type=int, default=4)  # 12k -> 27k/42k/57k/72k
    parser.add_argument("--chat-max-train", type=int, default=600, help="600 real (300 old + 300 new) + 4 refusal")
    parser.add_argument("--n-old-real", type=int, default=300)
    parser.add_argument("--n-generation-prompts", type=int, default=30)
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
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/local/hz_world_run2_fork_matched_chat"))
    parser.add_argument("--results-file", type=Path,
                         default=Path("results/local/hz_world_run2_fork_matched_chat.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    print(f"[fork-D] loading base checkpoint: {args.base_checkpoint}", flush=True)
    model.load_state_dict(torch.load(args.base_checkpoint, map_location="cpu"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[fork-D] ARM D -- moderate breadth, MATCHED exposure: n_params={n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    with open(args.base_results) as f:
        base_results = json.load(f)
    base_step = base_results["total_steps"]
    mastery = base_results["final_mastery"]
    base_calls = base_results["per_stage_calls"]

    squad_train, squad_held, squad_paraphrase = build_squad_split(seed=args.seed)
    squad_train_contexts = sorted(set(item["context"] for item in squad_train))
    squad_held_contexts = sorted(set(item["context"] for item in squad_held))
    chat_train, chat_held = build_chat_split(seed=args.seed, max_train=args.chat_max_train, max_held_out=50)
    assert len(chat_held) == 51, f"held-out set size changed ({len(chat_held)} != 51)"

    n_real = args.chat_max_train
    real_part, refusal_part = chat_train[:n_real], chat_train[n_real:]
    old_pool = real_part[:args.n_old_real] + refusal_part  # byte-identical to Arm A's exact 304
    new_pool = real_part[args.n_old_real:]
    old_baseline_exposure = base_calls["Chat"] / len(old_pool)
    print(f"[fork-D] old pool: {len(old_pool)} (baked-in avg exposure {old_baseline_exposure:.2f} from "
          f"{base_calls['Chat']} Chat calls at step {base_step}), new pool: {len(new_pool)}, "
          f"combined: {len(old_pool) + len(new_pool)}", flush=True)

    chat_rng = random.Random(args.seed + 52)
    chat_pick, chat_state = make_chat_picker(old_pool, new_pool, old_baseline_exposure, chat_rng)

    print(f"[fork-D] real data: {len(squad_train_contexts)} corpus paragraphs, {len(TRAIN_FACTS)} knowledge facts, "
          f"{len(chat_train)} chat train (Arm A had 304, Arm C had 1741), {len(chat_held)} chat held-out",
          flush=True)

    wordlist = load_wordlist()

    stage_order = ["Corpus", "L0", "L1", "L2", "L3", "L4-logic", "L4-counting", "L5", "L6",
                   "Library", "Knowledge", "Chat"]
    rngs = {s: random.Random(args.seed + i) for i, s in enumerate(stage_order)}
    corpus_rng = random.Random(args.seed + 50)
    knowledge_rng = random.Random(args.seed + 51)
    subskill_eval_rngs = {"L3": random.Random(args.seed + 4 + TEST_SEED_OFFSET),
                           "L4-logic": random.Random(args.seed + 5 + TEST_SEED_OFFSET)}
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
        "Chat": lambda: chat_train_step(model, opt, tok, chat_pick()),
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
    per_stage_calls = {s: 0 for s in stage_order}
    total_steps = args.steps_per_eval_point * args.n_eval_points

    eval_points = []
    t0 = time.time()
    for step in range(total_steps):
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
            print(f"[fork-D] cumulative_step={cumulative_step} (+{step+1}/{total_steps}, {elapsed:.0f}s, "
                  f"{(step+1)/elapsed:.2f} steps/sec) last={chosen} chat_phase={chat_state['phase']} "
                  f"chat_calls_this_phase={per_stage_calls['Chat']}", flush=True)

        if (step + 1) % args.steps_per_eval_point == 0:
            ckpt_path = args.checkpoint_dir / f"step_{cumulative_step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[fork-D] checkpoint saved: {ckpt_path}", flush=True)

            total_chat_exposure = base_calls["Chat"] + per_stage_calls["Chat"]
            combined_pool_size = len(old_pool) + len(new_pool)
            print(f"\n[fork-D] ===== RETENTION @ cumulative_step={cumulative_step} "
                  f"(avg Chat exposure so far: {total_chat_exposure / combined_pool_size:.2f}/example, "
                  f"target ~16) =====", flush=True)
            final_scores = {}
            for s in stage_order:
                try:
                    final_scores[s] = retention_fns[s](model)
                    print(f"[fork-D] {s}: {final_scores[s]}", flush=True)
                except Exception as e:
                    print(f"[fork-D] {s}: eval error {e}", flush=True)
                    final_scores[s] = {"error": str(e)}

            print(f"\n[fork-D] ===== CHAT GENERATION @ cumulative_step={cumulative_step} "
                  f"({args.n_generation_prompts} fixed held-out prompts) =====", flush=True)
            samples = sample_generations(model, tok, chat_held, n=args.n_generation_prompts)
            for s in samples[:6]:
                print(f"[fork-D] Q: {s['instruction']}", flush=True)
                print(f"[fork-D]   generated: {s['generated']!r}", flush=True)
            diversity = chat_diversity_metrics(samples)
            coherence = coherent_english_rate(samples, wordlist)
            relevance = semantic_relevance_heuristic(samples, wordlist)
            print(f"[fork-D] chat diversity: {diversity}", flush=True)
            print(f"[fork-D] coherent_english_rate (heuristic, known-gameable): {coherence}", flush=True)
            print(f"[fork-D] semantic_relevance_heuristic (weak proxy): {relevance}", flush=True)

            eval_points.append({
                "cumulative_step": cumulative_step,
                "avg_chat_exposure_per_example": total_chat_exposure / combined_pool_size,
                "chat_sampler_state": dict(chat_state),
                "final_scores": final_scores,
                "chat_diversity": diversity,
                "coherent_english_rate": coherence,
                "semantic_relevance_heuristic": relevance,
                "samples": samples,
                "calls_this_phase": dict(per_stage_calls),
                "elapsed_seconds": time.time() - t0,
            })
            with open(args.results_file, "w") as f:
                json.dump({"base_checkpoint": str(args.base_checkpoint), "base_step": base_step,
                            "n_params": n_params, "old_pool_size": len(old_pool), "new_pool_size": len(new_pool),
                            "old_baseline_exposure": old_baseline_exposure, "eval_points": eval_points},
                          f, indent=2)
            print(f"[fork-D] wrote {args.results_file} ({len(eval_points)} eval points so far)\n", flush=True)

    total_time = time.time() - t0
    print(f"\n[fork-D] fork done: {total_steps} steps in {total_time:.0f}s "
          f"(cumulative_step={base_step + total_steps})", flush=True)
    print(f"[fork-D] DONE.", flush=True)


if __name__ == "__main__":
    main()
