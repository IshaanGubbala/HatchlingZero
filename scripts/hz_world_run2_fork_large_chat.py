#!/usr/bin/env python3
"""Hatchling World Run 2 fork -- Arm C: full-HW continuation with a much
LARGER real Chat pool (plans/Hatchling world.md section 0.6), user-
directed. Context: the interference fork (Arm A vs Arm B,
`hz_world_run2_fork_chat_only.py`, already run) ruled out gradient
interference as the reason Chat can't generate coherently -- protected
Chat-only training bought a small loss edge but WORSE generation and
catastrophic collateral damage. The remaining live hypothesis: Chat's
300-example train pool is simply too small/undiverse for a 5M-param
model to learn general conversational behavior, independent of
scheduling.

Arm C starts from the EXACT SAME `step_12000.pt` checkpoint as Arm A
and Arm B, runs the SAME full 12-channel interleaved scheduler for the
SAME 24,000-step budget (18k/24k/30k/36k cumulative, identical to
Arm A), with everything held fixed except the Chat train pool size:
300 -> ~1,741 (the full usable Dolly open_qa/general_qa pool at this
session's length filter). Verified before launch, not assumed: the
held-out set and the fixed generation prompts are governed by
`build_chat_split`'s `n_held = round(len(items) * HELD_OUT_FRAC)`,
computed from the FULL item list before any `max_train` cap is
applied -- so raising `max_train` does not touch which 51 examples are
held out or which prompts are "fixed." Confirmed by direct comparison:
the held-out set and the first 6 prompts are IDENTICAL between
max_train=300 and max_train=None at the same seed.

Also expands the fixed generation-prompt set from Arm A/B's 6 to 30
(`--n-generation-prompts`), per the user's explicit concern that 6
prompts risk one lucky generation. Adds two automated HEURISTIC
proxies, disclosed as proxies, not ground truth:
  - coherent_english_rate: fraction of a generation's whitespace
    tokens that are real English dictionary words (from
    `/usr/share/dict/words` if present, else a small built-in
    fallback list -- real, disclosed portability fallback for
    machines without that file). Not a grammaticality judgment, just
    a real-word-ratio proxy.
  - semantic_relevance_heuristic: whether a generation shares any
    real-English content word (len>3, not a stopword) with its
    prompt. A weak lower bound, NOT a real semantic-correctness
    judgment -- true semantic-relevance labeling on the final
    checkpoint's samples is done by manual read after this run
    completes, exactly as the user said is acceptable at this scale.
"""
from __future__ import annotations

import argparse
import json
import random
import re
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
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.language.tokenizer import NOVEL_LABELS  # noqa: E402
from hatchling_world.knowledge.facts import TRAIN_FACTS  # noqa: E402
from hatchling_world.knowledge.chat_data import build_chat_split  # noqa: E402
from hatchling_world.knowledge.squad_corpus import build_squad_split  # noqa: E402

TEST_SEED_OFFSET = nt.TEST_SEED_OFFSET

_FALLBACK_WORDS = set("""
the a an and or but in on at to of for with is are was were be been being
have has had do does did will would could should may might must can cannot
this that these those i you he she it we they them him her his hers its our
ours your yours their theirs what which who whom whose where when why how
not no yes all any some few more most other such only own same so than too
very just also here there then now up down out off over under again further
once about above below between into through during before after from as
because if while when than not never always often sometimes usually people
person like know think want good bad big small new old first last long
great little own other come go get make see look use find give tell ask
work seem feel try leave call
""".split())


def load_wordlist() -> set:
    for path in (Path("/usr/share/dict/words"), Path("/usr/dict/words")):
        try:
            with open(path) as f:
                return {w.strip().lower() for w in f if w.strip().isalpha()}
        except OSError:
            continue
    return _FALLBACK_WORDS


def coherent_english_rate(samples: list, wordlist: set, min_frac: float = 0.5) -> dict:
    def frac_real_words(s: str) -> float:
        toks = re.findall(r"[a-zA-Z']+", s)
        if not toks:
            return 0.0
        return sum(1 for t in toks if t.lower() in wordlist) / len(toks)

    fracs = [frac_real_words(s["generated"]) for s in samples]
    return {"rate_ge_50pct_real_words": sum(1 for f in fracs if f >= min_frac) / len(fracs),
            "avg_real_word_fraction": sum(fracs) / len(fracs)}


def semantic_relevance_heuristic(samples: list, wordlist: set) -> dict:
    def content_words(s: str) -> set:
        toks = re.findall(r"[a-zA-Z']+", s.lower())
        return {t for t in toks if t in wordlist and t not in _FALLBACK_WORDS and len(t) > 3}

    hits = [len(content_words(s["instruction"]) & content_words(s["generated"])) > 0 for s in samples]
    return {"keyword_overlap_rate": sum(hits) / len(hits)}


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
    parser.add_argument("--n-eval-points", type=int, default=4)  # 12k -> 18k/24k/30k/36k, matches Arm A exactly
    parser.add_argument("--chat-max-train", type=int, default=None,
                         help="None = full usable Dolly pool (~1,741); Arm A used 300")
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
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/local/hz_world_run2_fork_large_chat"))
    parser.add_argument("--results-file", type=Path,
                         default=Path("results/local/hz_world_run2_fork_large_chat.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    print(f"[fork-C] loading base checkpoint: {args.base_checkpoint}", flush=True)
    model.load_state_dict(torch.load(args.base_checkpoint, map_location="cpu"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[fork-C] ARM C -- full-HW continuation, LARGER Chat pool: n_params={n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    with open(args.base_results) as f:
        base_results = json.load(f)
    base_step = base_results["total_steps"]
    mastery = base_results["final_mastery"]
    base_calls = base_results["per_stage_calls"]
    print(f"[fork-C] warm-started mastery + calls from step {base_step} (same as Arm A)", flush=True)

    squad_train, squad_held, squad_paraphrase = build_squad_split(seed=args.seed)
    squad_train_contexts = sorted(set(item["context"] for item in squad_train))
    squad_held_contexts = sorted(set(item["context"] for item in squad_held))
    chat_train, chat_held = build_chat_split(seed=args.seed, max_train=args.chat_max_train, max_held_out=50)
    print(f"[fork-C] real data: {len(squad_train_contexts)} corpus paragraphs, {len(TRAIN_FACTS)} knowledge facts, "
          f"{len(chat_train)} CHAT TRAIN (Arm A had 304), {len(chat_held)} chat held-out "
          f"(must match Arm A's 51 -- verified pre-launch)", flush=True)
    assert len(chat_held) == 51, f"held-out set size changed ({len(chat_held)} != 51) -- would break comparability"

    wordlist = load_wordlist()
    print(f"[fork-C] coherence/relevance wordlist size: {len(wordlist)}", flush=True)

    stage_order = ["Corpus", "L0", "L1", "L2", "L3", "L4-logic", "L4-counting", "L5", "L6",
                   "Library", "Knowledge", "Chat"]
    rngs = {s: random.Random(args.seed + i) for i, s in enumerate(stage_order)}
    corpus_rng = random.Random(args.seed + 50)
    knowledge_rng = random.Random(args.seed + 51)
    chat_rng = random.Random(args.seed + 52)
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
            print(f"[fork-C] cumulative_step={cumulative_step} (+{step+1}/{total_steps}, {elapsed:.0f}s, "
                  f"{(step+1)/elapsed:.2f} steps/sec) last={chosen} calls_this_phase={per_stage_calls}", flush=True)

        if (step + 1) % args.steps_per_eval_point == 0:
            ckpt_path = args.checkpoint_dir / f"step_{cumulative_step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[fork-C] checkpoint saved: {ckpt_path}", flush=True)

            print(f"\n[fork-C] ===== RETENTION @ cumulative_step={cumulative_step} =====", flush=True)
            final_scores = {}
            for s in stage_order:
                try:
                    final_scores[s] = retention_fns[s](model)
                    print(f"[fork-C] {s}: {final_scores[s]}", flush=True)
                except Exception as e:
                    print(f"[fork-C] {s}: eval error {e}", flush=True)
                    final_scores[s] = {"error": str(e)}

            print(f"\n[fork-C] ===== CHAT GENERATION @ cumulative_step={cumulative_step} "
                  f"({args.n_generation_prompts} fixed held-out prompts) =====", flush=True)
            samples = sample_generations(model, tok, chat_held, n=args.n_generation_prompts)
            for s in samples[:6]:  # print only the first 6 to keep the log readable; all are saved to JSON
                print(f"[fork-C] Q: {s['instruction']}", flush=True)
                print(f"[fork-C]   generated: {s['generated']!r}", flush=True)
            diversity = chat_diversity_metrics(samples)
            coherence = coherent_english_rate(samples, wordlist)
            relevance = semantic_relevance_heuristic(samples, wordlist)
            print(f"[fork-C] chat diversity: {diversity}", flush=True)
            print(f"[fork-C] coherent_english_rate (heuristic): {coherence}", flush=True)
            print(f"[fork-C] semantic_relevance_heuristic (weak proxy, real manual read needed): {relevance}",
                  flush=True)

            cumulative_calls = {s: base_calls[s] + per_stage_calls[s] for s in stage_order}
            eval_points.append({
                "cumulative_step": cumulative_step,
                "final_scores": final_scores,
                "chat_diversity": diversity,
                "coherent_english_rate": coherence,
                "semantic_relevance_heuristic": relevance,
                "samples": samples,
                "calls_this_phase": dict(per_stage_calls),
                "calls_cumulative": cumulative_calls,
                "elapsed_seconds": time.time() - t0,
            })
            with open(args.results_file, "w") as f:
                json.dump({"base_checkpoint": str(args.base_checkpoint), "base_step": base_step,
                            "n_params": n_params, "chat_train_size": len(chat_train),
                            "eval_points": eval_points}, f, indent=2)
            print(f"[fork-C] wrote {args.results_file} ({len(eval_points)} eval points so far)\n", flush=True)

    total_time = time.time() - t0
    print(f"\n[fork-C] fork done: {total_steps} steps in {total_time:.0f}s "
          f"(cumulative_step={base_step + total_steps})", flush=True)
    print(f"[fork-C] DONE.", flush=True)


if __name__ == "__main__":
    main()
